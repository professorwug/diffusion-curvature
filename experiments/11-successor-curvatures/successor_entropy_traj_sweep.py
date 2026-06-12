"""Focused sweep: Successor Entropy on τ-Colosseum, n_samples=2000 fixed,
vary n_trajectories, for softmax τ ∈ {0.2, 0.3, 0.5}.

For each n_trajectories ∈ {50, 100, 200, 500, 1000, 2000, 5000}:
  - Train FB once per manifold on that many trajectories (subset of a single
    master trajectory pool, to keep manifolds + trajectory seeds identical
    across n_trajectories).
  - Apply SuccessorEntropyCurvature at each softmax τ (post-training,
    ~free) and store the per-manifold mean ⟨ks_hat⟩ on visited nodes.

Writes to `processed_data/successor_entropy_traj_sweep.csv`. A proper
method should yield Pearson(⟨ks_hat⟩, ks_true) that *grows* with n_traj
and is *consistent* across τ.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd

from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum


N_SAMPLES = 2000
TRAJ_LENGTH = 50
NUM_MANIFOLDS = 20
N_TRAJ_GRID_DEFAULT = [50, 100, 200, 500, 1000, 2000, 5000]
TAUS = [0.2, 0.3, 0.5]

FB_KW_DEFAULT = dict(z_dim=2, hidden_dim=256, gamma=0.5, n_epochs=150)


def _pick_torch_device(req: str) -> str:
    import torch
    if req and req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


def train_and_measure(
    X: np.ndarray,
    trajectories: np.ndarray,
    device: str,
    seed: int,
    fb_kw: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train FB on the given trajectories, return (M, F_emb, B_emb) on corpus."""
    trainer = FBTrainer(
        obs_dim=X.shape[1],
        **fb_kw,
        batch_size=min(512, max(8, trajectories.shape[0] * trajectories.shape[1] // 4)),
        device=device,
        seed=seed,
    )
    trainer.fit(trajectories)
    return compute_successor_measures(
        trainer.F_net, trainer.B_net, X, X, device=device,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="processed_data/successor_entropy_traj_sweep.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--gamma", type=float, default=FB_KW_DEFAULT["gamma"])
    p.add_argument("--z-dim", type=int, default=FB_KW_DEFAULT["z_dim"])
    p.add_argument("--hidden-dim", type=int, default=FB_KW_DEFAULT["hidden_dim"])
    p.add_argument("--n-epochs", type=int, default=FB_KW_DEFAULT["n_epochs"])
    p.add_argument("--n-trajs", default=",".join(str(n) for n in N_TRAJ_GRID_DEFAULT),
                   help="Comma-separated list of n_trajectories values to sweep")
    args = p.parse_args()

    device = _pick_torch_device(args.device)
    fb_kw = dict(z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                 gamma=args.gamma, n_epochs=args.n_epochs)
    n_traj_grid = [int(s) for s in args.n_trajs.split(",") if s.strip()]
    print(f"Torch device: {device}")
    print(f"FB: {fb_kw}")
    print(f"n_trajectories grid: {n_traj_grid}")

    # Build the master dataset once with the largest trajectory count; subset
    # trajectories for smaller n_traj conditions so the same manifolds + same
    # initial walks persist across n_traj.
    max_n_traj = max(n_traj_grid)
    print(f"Building master TauColosseum (n_samples={N_SAMPLES}, "
          f"n_trajectories={max_n_traj}, manifolds={NUM_MANIFOLDS})…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = TauColosseum(
            n_samples=N_SAMPLES,
            n_trajectories=max_n_traj,
            traj_length=TRAJ_LENGTH,
            intrinsic_dims=[2],
            codimensions=[1],
            noise_levels=[0.0],
            num_manifolds_per_dim=NUM_MANIFOLDS,
            knn=10,
            seed=args.seed,
            save_directory="/tmp/.tau-cc-successor-traj-sweep",
        )
    print(f"  built {len(cc)} manifolds")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Streaming write so partial runs are recoverable.
    header_written = out.exists() and out.stat().st_size > 0
    rows_written = 0

    for n_traj in n_traj_grid:
        print(f"\n=== n_trajectories = {n_traj} ===")
        for i in range(len(cc)):
            inst = cc.get_item(i)
            all_traj = np.asarray(inst["trajectories"])  # (max_n_traj, T+1, d)
            all_idx = np.asarray(inst["trajectories_idx"])
            traj_subset = all_traj[:n_traj]
            idx_subset = all_idx[:n_traj]
            visited = np.unique(idx_subset.ravel())
            ks_arr = np.asarray(cc.DS[i].obj["ks"], dtype=float)
            ks_true = float(ks_arr.mean()) if ks_arr.size > 1 else float(ks_arr.item())

            try:
                M, F_emb, B_emb = train_and_measure(
                    inst["X"], traj_subset, device=device,
                    seed=args.seed + i + 1000 * n_traj, fb_kw=fb_kw,
                )
                fb_err = ""
            except Exception as e:
                print(f"  [err] n_t={n_traj} inst={i} FB train: {e}")
                fb_err = str(e)[:200]
                for tau in TAUS:
                    pd.DataFrame([{
                        "n_trajectories": n_traj, "manifold": i, "tau": tau,
                        "ks_hat_mean": float("nan"), "ks_true": ks_true,
                        "n_visited": int(visited.size), "err": fb_err,
                    }]).to_csv(out, mode="a", index=False, header=not header_written)
                    header_written = True
                    rows_written += 1
                continue

            for tau in TAUS:
                sec = SuccessorEntropyCurvature(tau=tau)
                k_hat = np.asarray(
                    sec.fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb),
                    dtype=float,
                )
                k_vis = k_hat[visited]
                k_vis = k_vis[np.isfinite(k_vis)]
                mean_hat = float(k_vis.mean()) if k_vis.size else float("nan")
                pd.DataFrame([{
                    "n_trajectories": n_traj, "manifold": i, "tau": tau,
                    "ks_hat_mean": mean_hat, "ks_true": ks_true,
                    "n_visited": int(visited.size), "err": "",
                }]).to_csv(out, mode="a", index=False, header=not header_written)
                header_written = True
                rows_written += 1
            print(f"  manifold {i+1:2d}/{len(cc)} ks_true={ks_true:+.3f} "
                  f"ks_hat(τ=0.2)={(mean_hat):+.3f}")

    print(f"\nWrote {rows_written} rows → {out}")


if __name__ == "__main__":
    main()
