"""Round 2: keep z_dim=8 + corpus aggregation (the round-1 winner), and
push the FB training budget to see if we can lift ρ from 0.65 → ≥ 0.8 at
d=2.

Configs swept (all fixed: z_dim=8, hidden=256, gamma=0.9, batch=1024,
cosine_lr, lr=1e-4, agg=corpus):

    name                    n_traj   n_epochs   ~per-FB time
    baseline (round 1)         200        300       18 s    (already done)
    more_epochs                200        600       35 s
    more_traj                  500        300       45 s
    more_both                  500        600       90 s
    all_in                     500       1200      180 s

20 manifolds × 2 noise levels × 5 configs = 200 trains.
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
CSV = OUT_DIR / "round2.csv"

DEVICE = os.environ.get("DEVICE", "cuda:1")

DIM = 2
NOISES = [0.05, 0.10]
NUM_MANIFOLDS = 20

# Each config tag dictates (n_trajectories, n_epochs). Other knobs fixed.
CONFIGS: list[tuple[str, int, int]] = [
    ("baseline_n200_e300",  200,  300),
    ("more_epochs_n200_e600", 200, 600),
    ("more_traj_n500_e300",   500, 300),
    ("more_both_n500_e600",   500, 600),
    ("all_in_n500_e1200",     500, 1200),
]


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def existing_keys() -> set[tuple[str, float, int]]:
    if not CSV.exists() or CSV.stat().st_size == 0:
        return set()
    df = pd.read_csv(CSV, usecols=["config", "noise", "instance"])
    return set(zip(df["config"], df["noise"].astype(float),
                    df["instance"].astype(int)))


def append_row(row: dict) -> None:
    header = not CSV.exists() or CSV.stat().st_size == 0
    pd.DataFrame([row]).to_csv(CSV, mode="a", index=False, header=header)


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Output: {CSV}")
    print(f"Configs: {[c[0] for c in CONFIGS]}")
    done = existing_keys()
    if done:
        print(f"Resume: {len(done)} (config, noise, instance) triples already saved.")

    # Cache TauColosseums by (n_samples, n_trajectories, traj_length, noise).
    # n_samples and traj_length are fixed; only n_traj varies across configs.
    dataset_cache: dict[tuple[int, float], TauColosseum] = {}

    def get_cc(n_traj: int, noise: float) -> TauColosseum:
        key = (n_traj, noise)
        if key not in dataset_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset_cache[key] = TauColosseum(
                    n_samples=2000, n_trajectories=n_traj, traj_length=50,
                    intrinsic_dims=[DIM], codimensions=[1],
                    noise_levels=[noise], num_manifolds_per_dim=NUM_MANIFOLDS,
                    knn=10, seed=2024,
                    save_directory=f"/tmp/.tau-r2-d{DIM}-n{noise:g}-T{n_traj}",
                )
        return dataset_cache[key]

    for cfg_name, n_traj, n_epochs in CONFIGS:
        for noise in NOISES:
            cc = get_cc(n_traj, noise)
            print(f"\n--- {cfg_name}  ε={noise:g}  ({len(cc)} manifolds) ---",
                  flush=True)
            for i in range(len(cc)):
                if (cfg_name, float(noise), i) in done:
                    continue
                inst = cc.get_item(i)
                ks_true = float(np.asarray(inst["ks"]).mean())
                t0 = time.time()
                trainer = FBTrainer(
                    obs_dim=inst["X"].shape[1],
                    z_dim=8, hidden_dim=256, gamma=0.9,
                    n_epochs=n_epochs, cosine_lr=True,
                    batch_size=1024, lr=1e-4,
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
                    "config": cfg_name, "n_trajectories": n_traj,
                    "n_epochs": n_epochs, "noise": noise, "instance": i,
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
    print("\n========== Pearson r per (config × ε) — d=2 ==========")
    print(f"{'config':>30}  {'ε=0.05':>9}  {'ε=0.10':>9}  {'mean':>9}")
    for cfg_name, n_traj, n_epochs in CONFIGS:
        rs = []
        for noise in NOISES:
            cell = df[(df.config == cfg_name) & np.isclose(df.noise, noise)]
            r = pearson(cell.ks_hat, cell.ks_true)
            rs.append(r)
        print(f"{cfg_name:>30}  {rs[0]:+9.3f}  {rs[1]:+9.3f}  "
              f"{np.nanmean(rs):+9.3f}")


if __name__ == "__main__":
    main()
