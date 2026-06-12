"""Torus FB replication — sanity check for Successor Entropy on a clean manifold.

Replicates the experiment from [[Epistemic Empowerment via Learning Successor
Functions]] §"Do Successor Measures recover geometry of a sparsely-sampled
torus better than vanilla transition modeling":

    torus(n=2000) → kNN graph (k=10) → sample N walks of T steps
      → train FB (z=2) → reconstruct P̂^(γ) = rownorm(max(0, F·B^T))
      → compare to real P^(γ) = (I - γP)^(-1) (row-normalized)
      → row-wise KL(real || reconstructed), mean
      → row entropy of P̂^(γ), correlate with Gaussian curvature ks

Sweep: N ∈ {10, 20, 50, 100}, T ∈ {20, 50}, γ ∈ {0.2, 0.5, 0.8, 0.9}, 3 seeds.
Writes `processed_data/torus_fb_replication.csv` incrementally.
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
import pygsp
import scipy.stats

from diffusion_curvature.datasets import torus
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories


N_POINTS = 2000
KNN = 10
Z_DIM = 2
HIDDEN_DIM = 256
N_EPOCHS = 150
N_DEFAULT = [10, 20, 50, 100]
T_DEFAULT = [20, 50]
GAMMA_DEFAULT = [0.2, 0.5, 0.8, 0.9]
SEEDS_DEFAULT = [0, 1, 2]


def _row_normalize(W) -> np.ndarray:
    import scipy.sparse as sp
    if sp.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)
    s = W.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return W / s


def true_resolvent(P: np.ndarray, gamma: float) -> np.ndarray:
    n = P.shape[0]
    return np.linalg.solve(np.eye(n) - gamma * P, np.eye(n))


def row_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return (p * (np.log(p) - np.log(q))).sum(axis=1)


def row_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, None)
    return -(p * np.log(p)).sum(axis=1)


def _pick_torch_device(req: str) -> str:
    import torch
    if req and req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="processed_data/torus_fb_replication.csv")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-walks", default=",".join(str(n) for n in N_DEFAULT))
    p.add_argument("--traj-lengths", default=",".join(str(t) for t in T_DEFAULT))
    p.add_argument("--gammas", default=",".join(str(g) for g in GAMMA_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    p.add_argument("--z-dim", type=int, default=Z_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    args = p.parse_args()

    device = _pick_torch_device(args.device)
    n_walks = [int(s) for s in args.n_walks.split(",") if s.strip()]
    traj_lengths = [int(s) for s in args.traj_lengths.split(",") if s.strip()]
    gammas = [float(s) for s in args.gammas.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Torch device: {device}")
    print(f"N walks: {n_walks} | T: {traj_lengths} | γ: {gammas} | seeds: {seeds}")
    print(f"FB: z_dim={args.z_dim}, hidden_dim={args.hidden_dim}, n_epochs={args.n_epochs}")

    print(f"Sampling torus ({N_POINTS} pts)…")
    X, ks = torus(n=N_POINTS, seed=0)
    X = np.asarray(X, dtype=np.float32)
    ks = np.asarray(ks, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = pygsp.graphs.NNGraph(X, k=KNN)
    P = _row_normalize(G.W)  # dense (2000x2000) — fine

    # Cache real resolvents per γ
    real_resolvents: dict[float, np.ndarray] = {}
    real_entropies: dict[float, np.ndarray] = {}
    for g in gammas:
        Rg = true_resolvent(P, g)
        Rg_norm = _row_normalize(Rg)
        real_resolvents[g] = Rg_norm
        real_entropies[g] = row_entropy(Rg_norm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_written = out.exists() and out.stat().st_size > 0
    rows_written = 0

    for seed in seeds:
        for T in traj_lengths:
            for N in n_walks:
                # Sample N walks of length T from P. Use per-(seed, T, N) rng,
                # and (critically) reuse prefixes where possible so smaller N is
                # a subset of larger N — stable within a (seed, T).
                max_N = max(n_walks)
                rng = np.random.default_rng(1000 * seed + T)
                all_traj_idx = subsample_trajectories(
                    G, n_trajectories=max_N, length=T, rng=rng,
                )
                traj_idx = all_traj_idx[:N]
                traj_coords = X[traj_idx]

                for gamma in gammas:
                    tag = f"seed={seed} T={T} N={N} γ={gamma}"
                    try:
                        trainer = FBTrainer(
                            obs_dim=X.shape[1],
                            z_dim=args.z_dim,
                            hidden_dim=args.hidden_dim,
                            gamma=gamma,
                            n_epochs=args.n_epochs,
                            batch_size=min(512, max(8, N * T // 4)),
                            device=device,
                            seed=seed,
                        )
                        trainer.fit(traj_coords, log_every=None)
                        M, _, _ = compute_successor_measures(
                            trainer.F_net, trainer.B_net, X, X, device=device,
                        )
                        P_hat = _row_normalize(M)
                        # Entropy correlation
                        H_hat = row_entropy(P_hat)
                        H_real = real_entropies[gamma]
                        mask = np.isfinite(H_hat) & np.isfinite(ks)
                        r_pearson_ks = scipy.stats.pearsonr(H_hat[mask], ks[mask])[0]
                        r_spearman_ks = scipy.stats.spearmanr(H_hat[mask], ks[mask])[0]
                        r_pearson_real = scipy.stats.pearsonr(H_hat[mask], H_real[mask])[0]
                        # Mean row KL (real || reconstructed)
                        kls = row_kl(real_resolvents[gamma], P_hat)
                        mean_kl = float(np.nanmean(kls))

                        pd.DataFrame([{
                            "seed": seed, "T": T, "N": N, "gamma": gamma,
                            "mean_kl": mean_kl,
                            "pearson_entropy_vs_ks": float(r_pearson_ks),
                            "spearman_entropy_vs_ks": float(r_spearman_ks),
                            "pearson_entropy_hat_vs_real": float(r_pearson_real),
                            "err": "",
                        }]).to_csv(out, mode="a", index=False, header=not header_written)
                        header_written = True
                        rows_written += 1
                        print(f"  {tag}: KL={mean_kl:.3f} "
                              f"pearson(H_hat, ks)={r_pearson_ks:+.3f} "
                              f"pearson(H_hat, H_real)={r_pearson_real:+.3f}")
                    except Exception as e:
                        msg = str(e)[:200]
                        print(f"  [err] {tag}: {msg}")
                        pd.DataFrame([{
                            "seed": seed, "T": T, "N": N, "gamma": gamma,
                            "mean_kl": float("nan"),
                            "pearson_entropy_vs_ks": float("nan"),
                            "spearman_entropy_vs_ks": float("nan"),
                            "pearson_entropy_hat_vs_real": float("nan"),
                            "err": msg,
                        }]).to_csv(out, mode="a", index=False, header=not header_written)
                        header_written = True
                        rows_written += 1

    print(f"\nWrote {rows_written} rows → {out}")


if __name__ == "__main__":
    main()
