"""v2 Successor Entropy sweep on τ-Colosseum with the torus-verified FB recipe.

Settings (from torus-v2 sweep that hit Spearman +0.54):
  z_dim=16, hidden_dim=256, n_epochs=600, cosine_lr=True, batch_size=1024
  softmax(M/τ) normalization with τ ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 5.0}
  laziness = Σ_j p_ij log p_ij  (Ollivier convention)

Sweep: γ=0.9 fixed, n_samples=2000, traj_length=50, 20 manifolds (d=2, c=1, noise=0),
n_trajectories ∈ {50, 100, 200, 500, 1000, 2000}. For each config + softmax τ,
record per-manifold mean laziness on visited nodes → computes Spearman across
manifolds against ks_true.

Writes `processed_data/successor_entropy_traj_sweep_v2.csv`.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd
import torch

from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum


N_SAMPLES = 2000
TRAJ_LENGTH = 50
NUM_MANIFOLDS = 20
N_TRAJ_GRID_DEFAULT = [50, 100, 200, 500, 1000, 2000]
TEMPS_DEFAULT = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

FB_KW = dict(
    z_dim=16, hidden_dim=256, n_epochs=600,
    cosine_lr=True, batch_size=1024,
)


def _pick_torch_device(req: str) -> str:
    if req and req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


def _raw_fb_matrix(trainer: FBTrainer, corpus: np.ndarray, device) -> np.ndarray:
    trainer.F_net.eval()
    trainer.B_net.eval()
    with torch.no_grad():
        x = torch.as_tensor(corpus, dtype=torch.float32, device=device)
        F1, F2 = trainer.F_net(x)
        B = trainer.B_net(x)
        M = torch.minimum(F1 @ B.T, F2 @ B.T)
    return M.cpu().numpy()


def row_softmax(M: np.ndarray, temperature: float) -> np.ndarray:
    x = M / max(temperature, 1e-12)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def entropic_laziness(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.clip(P, eps, None)
    return (P * np.log(P)).sum(axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="processed_data/successor_entropy_traj_sweep_v2.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--n-trajs", default=",".join(str(n) for n in N_TRAJ_GRID_DEFAULT))
    p.add_argument("--temps", default=",".join(str(t) for t in TEMPS_DEFAULT))
    p.add_argument("--n-epochs", type=int, default=FB_KW["n_epochs"])
    p.add_argument("--z-dim", type=int, default=FB_KW["z_dim"])
    p.add_argument("--hidden-dim", type=int, default=FB_KW["hidden_dim"])
    p.add_argument("--batch-size", type=int, default=FB_KW["batch_size"])
    args = p.parse_args()

    device = _pick_torch_device(args.device)
    n_traj_grid = [int(s) for s in args.n_trajs.split(",") if s.strip()]
    temps = [float(s) for s in args.temps.split(",") if s.strip()]
    fb_kw = dict(
        z_dim=args.z_dim, hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs, cosine_lr=True, batch_size=args.batch_size,
    )
    print(f"Torch device: {device}")
    print(f"FB: {fb_kw} | γ={args.gamma}")
    print(f"n_trajectories grid: {n_traj_grid} | τ: {temps}")

    max_n_traj = max(n_traj_grid)
    print(f"Building master TauColosseum ({NUM_MANIFOLDS} manifolds, "
          f"n_samples={N_SAMPLES}, n_trajectories={max_n_traj})…")
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
    header_written = out.exists() and out.stat().st_size > 0
    rows_written = 0

    for n_traj in n_traj_grid:
        print(f"\n=== n_trajectories = {n_traj} ===")
        for i in range(len(cc)):
            inst = cc.get_item(i)
            all_traj = np.asarray(inst["trajectories"])
            all_idx = np.asarray(inst["trajectories_idx"])
            traj_subset = all_traj[:n_traj]
            idx_subset = all_idx[:n_traj]
            visited = np.unique(idx_subset.ravel())
            ks_arr = np.asarray(cc.DS[i].obj["ks"], dtype=float)
            ks_true = float(ks_arr.mean()) if ks_arr.size > 1 else float(ks_arr.item())

            try:
                trainer = FBTrainer(
                    obs_dim=inst["X"].shape[1],
                    gamma=args.gamma,
                    device=device,
                    seed=args.seed + i + 1000 * n_traj,
                    **fb_kw,
                )
                trainer.fit(traj_subset, log_every=None)
                M = _raw_fb_matrix(trainer, inst["X"], device=device)

                for temperature in temps:
                    P_hat = row_softmax(M, temperature)
                    lazi = entropic_laziness(P_hat)
                    lazi_vis = lazi[visited]
                    lazi_vis = lazi_vis[np.isfinite(lazi_vis)]
                    mean_hat = float(lazi_vis.mean()) if lazi_vis.size else float("nan")
                    pd.DataFrame([{
                        "n_trajectories": n_traj, "manifold": i,
                        "gamma": args.gamma, "temperature": temperature,
                        "ks_hat_mean": mean_hat, "ks_true": ks_true,
                        "n_visited": int(visited.size), "err": "",
                    }]).to_csv(out, mode="a", index=False, header=not header_written)
                    header_written = True
                    rows_written += 1
                print(f"  manifold {i+1:2d}/{len(cc)} ks_true={ks_true:+.3f} "
                      f"laz(τ=1.0)~{mean_hat:+.3f}")
            except Exception as e:
                msg = str(e)[:200]
                print(f"  [err] n_t={n_traj} inst={i}: {msg}")
                for temperature in temps:
                    pd.DataFrame([{
                        "n_trajectories": n_traj, "manifold": i,
                        "gamma": args.gamma, "temperature": temperature,
                        "ks_hat_mean": float("nan"), "ks_true": ks_true,
                        "n_visited": int(visited.size), "err": msg,
                    }]).to_csv(out, mode="a", index=False, header=not header_written)
                    header_written = True
                    rows_written += 1

    print(f"\nWrote {rows_written} rows → {out}")


if __name__ == "__main__":
    main()
