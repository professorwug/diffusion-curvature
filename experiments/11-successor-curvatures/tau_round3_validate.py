"""Round 3: validate the round-2 winner across the full noise grid and at
d=3.

Winning config (round 2): z_dim=8, hidden=256, gamma=0.9, lr=1e-4,
batch=1024, cosine_lr, n_trajectories=500, n_epochs=300, agg=corpus.

Sweep:
    d        ∈ {2, 3}
    ε        ∈ {0.01, 0.05, 0.10, 0.20}
    n_manifolds = 20

Total: 2 × 4 × 20 = 160 trains × ~45 s = ~2 h on the GPU.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import scipy.stats

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum

OUT_DIR = EXP_ROOT / "processed_data" / "agg_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV = OUT_DIR / "round3.csv"

DEVICE = os.environ.get("DEVICE", "cuda:1")

DIMS = [2, 3]
NOISES = [0.01, 0.05, 0.10, 0.20]
NUM_MANIFOLDS = 20

# Winning config from round 2 — fixed for the entire round-3 sweep.
FB_CFG = dict(
    z_dim=8, hidden_dim=256, gamma=0.9,
    n_epochs=300, cosine_lr=True, batch_size=1024, lr=1e-4,
)
DATASET_CFG = dict(
    n_samples=2000, n_trajectories=500, traj_length=50, knn=10,
)


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def existing_keys() -> set[tuple[int, float, int]]:
    if not CSV.exists() or CSV.stat().st_size == 0:
        return set()
    df = pd.read_csv(CSV, usecols=["dim", "noise", "instance"])
    return set(zip(df["dim"].astype(int), df["noise"].astype(float),
                    df["instance"].astype(int)))


def append_row(row: dict) -> None:
    header = not CSV.exists() or CSV.stat().st_size == 0
    pd.DataFrame([row]).to_csv(CSV, mode="a", index=False, header=header)


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"FB cfg: {FB_CFG}")
    print(f"Dataset: {DATASET_CFG}")
    print(f"Sweep: dims={DIMS}, noises={NOISES}, n_manifolds={NUM_MANIFOLDS}")
    done = existing_keys()
    if done:
        print(f"Resume: {len(done)} (dim, noise, instance) triples already saved.")

    for d in DIMS:
        for noise in NOISES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cc = TauColosseum(
                    intrinsic_dims=[d], codimensions=[1],
                    noise_levels=[noise], num_manifolds_per_dim=NUM_MANIFOLDS,
                    seed=2024,
                    save_directory=f"/tmp/.tau-r3-d{d}-n{noise:g}",
                    **DATASET_CFG,
                )
            print(f"\n--- d={d}  ε={noise:g}  ({len(cc)} manifolds) ---",
                  flush=True)
            for i in range(len(cc)):
                if (d, float(noise), i) in done:
                    continue
                inst = cc.get_item(i)
                ks_true = float(np.asarray(inst["ks"]).mean())
                t0 = time.time()
                trainer = FBTrainer(
                    obs_dim=inst["X"].shape[1], **FB_CFG,
                    device=DEVICE, seed=11 + i,
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
                ks_hat = float(k.mean()) if k.size else float("nan")
                row = {
                    "dim": d, "noise": noise, "instance": i,
                    "ks_true": ks_true, "ks_hat": ks_hat,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                append_row(row)
                print(
                    f"  inst {i+1:2d}/{len(cc)}  ks_true={ks_true:+7.3f}  "
                    f"ks_hat={ks_hat:+7.3f}   [{row['elapsed_s']}s]",
                    flush=True,
                )

    # Final Pearson summary.
    df = pd.read_csv(CSV)
    print("\n========== Pearson r per (d × ε) ==========")
    print(f"{'d':>3}  " + "  ".join(f"ε={n:>5g}" for n in NOISES) + "    mean")
    for d in DIMS:
        rs = []
        for noise in NOISES:
            cell = df[(df.dim == d) & np.isclose(df.noise, noise)]
            r = pearson(cell.ks_hat, cell.ks_true)
            rs.append(r)
        print(f"{d:>3}  " + "  ".join(f"{r:+7.3f}" for r in rs)
              + f"    {np.nanmean(rs):+7.3f}")


if __name__ == "__main__":
    main()
