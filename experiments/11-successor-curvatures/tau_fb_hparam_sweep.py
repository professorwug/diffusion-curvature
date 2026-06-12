"""Quick local sweep over FB hyperparameters on a small TauColosseum.

Pinned at d=2, ε=0.05 (the cell where the production sweep showed weakest
Successor recovery). For each (z_dim, n_epochs, lr, gamma) tuple we train
FB once per manifold and report Pearson r between the corpus-mean k_SR
and the manifold's true scalar curvature, across `NUM_MANIFOLDS=12`
manifolds.

The aim is to find hparams that lift the d=2 Pearson without rebuilding
the dataset.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd
import scipy.stats

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum

DEVICE = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu"

DIM = 2
NOISE = 0.05
NUM_MANIFOLDS = 12

# Trajectory budget kept fixed at the small-test scale so each FB run is
# fast on CPU. Once we know which hparams help, the production rerun on
# della (with the matched-iid trajectory budget) will validate them.
DATASET_KW = dict(
    n_samples=600,
    n_trajectories=120,
    traj_length=40,
    knn=10,
)

# Each variant overrides the v3 baseline FB config.
BASELINE = dict(
    z_dim=16, hidden_dim=256, gamma=0.9,
    n_epochs=300, cosine_lr=True, batch_size=512, lr=1e-4,
)

VARIANTS = [
    ("baseline (v3-style)",      {}),
    ("longer training",          {"n_epochs": 1200}),
    ("very long training",       {"n_epochs": 2400}),
    ("lower z_dim=8",            {"z_dim": 8}),
    ("lower z_dim=4",            {"z_dim": 4}),
    ("z_dim=8 + long",           {"z_dim": 8, "n_epochs": 1200}),
    ("z_dim=4 + long",           {"z_dim": 4, "n_epochs": 1200}),
    ("higher lr 3e-4",           {"lr": 3e-4}),
    ("higher lr 1e-3",           {"lr": 1e-3}),
    ("lower gamma 0.7",          {"gamma": 0.7}),
    ("higher gamma 0.95",        {"gamma": 0.95}),
    ("z=4 + lr=3e-4 + long",     {"z_dim": 4, "lr": 3e-4, "n_epochs": 1200}),
]


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Cell: d={DIM} ε={NOISE} n_manifolds={NUM_MANIFOLDS}")
    print(f"Dataset: {DATASET_KW}")
    print(f"Baseline FB: {BASELINE}")
    print()

    # Build dataset once.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = TauColosseum(
            intrinsic_dims=[DIM], codimensions=[1],
            noise_levels=[NOISE], num_manifolds_per_dim=NUM_MANIFOLDS,
            seed=2024, save_directory=f"/tmp/.tau-hparam-d{DIM}-n{NOISE:g}",
            **DATASET_KW,
        )
    print(f"Built {len(cc)} instances")
    insts = [cc.get_item(i) for i in range(len(cc))]
    ks_true = [float(np.asarray(it["ks"]).mean()) for it in insts]
    print(f"ks_true range: [{min(ks_true):.2f}, {max(ks_true):.2f}]")
    print()

    rows = []
    for name, override in VARIANTS:
        cfg = {**BASELINE, **override}
        print(f"--- {name}  cfg={override} ---", flush=True)
        t0 = time.time()
        ks_hat = []
        for i, inst in enumerate(insts):
            trainer = FBTrainer(
                obs_dim=inst["X"].shape[1],
                **cfg,
                device=DEVICE,
                seed=11 + i,
            )
            trainer.fit(inst["trajectories"].astype(np.float32))
            M, F_emb, B_emb = compute_successor_measures(
                trainer.F_net, trainer.B_net,
                inst["X"], inst["X"], device=DEVICE,
            )
            k = np.asarray(
                SuccessorEntropyCurvature().fit_transform(
                    M=M, F_embeddings=F_emb, B_embeddings=B_emb,
                ),
                dtype=float,
            )
            k = k[np.isfinite(k)]
            ks_hat.append(float(k.mean()) if k.size else float("nan"))
        r = pearson(ks_hat, ks_true)
        elapsed = time.time() - t0
        spread = float(np.std(ks_hat)) if all(np.isfinite(ks_hat)) else float("nan")
        print(
            f"  Pearson r = {r:+.3f}   "
            f"k_hat spread = {spread:.3f}   "
            f"elapsed = {elapsed:.1f}s",
            flush=True,
        )
        rows.append({
            "variant": name, "pearson": r, "k_hat_std": spread,
            "elapsed_s": round(elapsed, 1),
            **{f"cfg_{k}": v for k, v in cfg.items()},
        })

    df = pd.DataFrame(rows)
    out = EXP_ROOT / "processed_data" / "tau_fb_hparam_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\n=== Summary (by Pearson r) ===")
    df_sorted = df.sort_values("pearson", ascending=False)
    for _, r in df_sorted.iterrows():
        print(f"  r={r.pearson:+.3f}  spread={r.k_hat_std:.3f}  {r.variant}  ({r.elapsed_s:.0f}s)")


if __name__ == "__main__":
    main()
