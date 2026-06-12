"""Torus FB replication v2 — matched to the original reason_reckon recipe.

Updates vs v1:
- z_dim: 16 (was 2)
- n_epochs: 600, cosine_lr True (was 150, flat)
- batch_size: 1024 (was min(512, N·T/4))
- Ground-truth graph: graphtools adaptive kNN k=30, anisotropy=1 (was pygsp k=10)
- Walks sampled from the *same* graphtools P (matches original "graphtools" walk_method)
- Measure normalization: softmax(M/τ) with temperature sweep τ ∈ {0.1,0.3,0.5,1,2,5}
- Laziness = Σ P log P (Ollivier convention; concentrated → near 0)
- Writes per-temperature rows to the same CSV.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import graphtools as gt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats
import torch

from diffusion_curvature.datasets import torus
from diffusion_curvature.successor.train import FBTrainer


N_POINTS = 2000
GT_KNN = 30
Z_DIM = 16
HIDDEN_DIM = 256
N_EPOCHS = 600
BATCH_SIZE = 1024
N_DEFAULT = [10, 20, 50, 100]
T_DEFAULT = [20, 50]
GAMMA_DEFAULT = [0.2, 0.5, 0.8, 0.9]
SEEDS_DEFAULT = [0, 1, 2]
TEMPS_DEFAULT = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]


def compute_gt_transition(X: np.ndarray, knn: int = GT_KNN) -> np.ndarray:
    G = gt.Graph(X, knn=knn, use_pygsp=True, anisotropy=1)
    W = G.K
    W = np.asarray(W.todense()) if sp.issparse(W) else np.asarray(W)
    s = W.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return W / s


def sample_walks_from_P(
    P: np.ndarray, n_walks: int, walk_length: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_walks of `walk_length + 1` nodes each from row-stochastic P.

    Returns (n_walks, walk_length + 1) node-index array.
    """
    n = P.shape[0]
    out = np.empty((n_walks, walk_length + 1), dtype=np.int64)
    for i in range(n_walks):
        cur = int(rng.integers(0, n))
        out[i, 0] = cur
        for t in range(walk_length):
            cur = int(rng.choice(n, p=P[cur]))
            out[i, t + 1] = cur
    return out


def true_resolvent(P: np.ndarray, gamma: float) -> np.ndarray:
    n = P.shape[0]
    return np.linalg.solve(np.eye(n) - gamma * P, np.eye(n))


def row_normalize(A: np.ndarray) -> np.ndarray:
    s = A.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return A / s


def row_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return (p * (np.log(p) - np.log(q))).sum(axis=1)


def entropic_laziness(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ollivier's 'laziness': Σ_j p_ij log p_ij — concentrated rows near 0, spread rows negative."""
    P = np.clip(P, eps, None)
    return (P * np.log(P)).sum(axis=1)


def row_softmax(M: np.ndarray, temperature: float) -> np.ndarray:
    x = M / max(temperature, 1e-12)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _raw_fb_matrix(trainer: FBTrainer, X: np.ndarray, device) -> np.ndarray:
    trainer.F_net.eval()
    trainer.B_net.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        F1, F2 = trainer.F_net(x)
        B = trainer.B_net(x)
        M = torch.minimum(F1 @ B.T, F2 @ B.T)
    return M.cpu().numpy()


def _pick_torch_device(req: str) -> str:
    if req and req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="processed_data/torus_fb_replication_v2.csv")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-walks", default=",".join(str(n) for n in N_DEFAULT))
    p.add_argument("--traj-lengths", default=",".join(str(t) for t in T_DEFAULT))
    p.add_argument("--gammas", default=",".join(str(g) for g in GAMMA_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--temps", default=",".join(str(t) for t in TEMPS_DEFAULT))
    p.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    p.add_argument("--z-dim", type=int, default=Z_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args()

    device = _pick_torch_device(args.device)
    n_walks = [int(s) for s in args.n_walks.split(",") if s.strip()]
    traj_lengths = [int(s) for s in args.traj_lengths.split(",") if s.strip()]
    gammas = [float(s) for s in args.gammas.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    temps = [float(s) for s in args.temps.split(",") if s.strip()]
    print(f"Torch device: {device}")
    print(f"N walks: {n_walks} | T: {traj_lengths} | γ: {gammas} | seeds: {seeds}")
    print(f"τ: {temps}")
    print(f"FB: z_dim={args.z_dim}, hidden={args.hidden_dim}, "
          f"n_epochs={args.n_epochs}, batch={args.batch_size}, cosine_lr=True")

    print(f"Sampling torus ({N_POINTS} pts) + graphtools P (k={GT_KNN})…")
    X, ks = torus(n=N_POINTS, seed=0)
    X = np.asarray(X, dtype=np.float32)
    ks = np.asarray(ks, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P_gt = compute_gt_transition(X, knn=GT_KNN)

    real_resolvents: dict[float, np.ndarray] = {}
    real_entropies: dict[float, np.ndarray] = {}  # Σ P log P, Ollivier's laziness
    for g in gammas:
        Rn = row_normalize(true_resolvent(P_gt, g))
        real_resolvents[g] = Rn
        real_entropies[g] = entropic_laziness(Rn)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_written = out.exists() and out.stat().st_size > 0
    rows_written = 0

    for seed in seeds:
        for T in traj_lengths:
            max_N = max(n_walks)
            rng = np.random.default_rng(1000 * seed + T)
            all_traj = sample_walks_from_P(P_gt, max_N, T, rng)
            for N in n_walks:
                traj_idx = all_traj[:N]
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
                            batch_size=args.batch_size,
                            cosine_lr=True,
                            device=device,
                            seed=seed,
                        )
                        trainer.fit(traj_coords, log_every=None)
                        M = _raw_fb_matrix(trainer, X, device=device)

                        best_sp = -np.inf
                        best_temp = None
                        for temperature in temps:
                            P_hat = row_softmax(M, temperature)
                            H_hat = entropic_laziness(P_hat)
                            mask = np.isfinite(H_hat) & np.isfinite(ks)
                            r_p_ks = float(scipy.stats.pearsonr(H_hat[mask], ks[mask])[0])
                            r_s_ks = float(scipy.stats.spearmanr(H_hat[mask], ks[mask])[0])
                            r_p_real = float(scipy.stats.pearsonr(
                                H_hat[mask], real_entropies[gamma][mask])[0])
                            kls = row_kl(real_resolvents[gamma], P_hat)
                            mean_kl = float(np.nanmean(kls))

                            pd.DataFrame([{
                                "seed": seed, "T": T, "N": N, "gamma": gamma,
                                "temperature": temperature,
                                "mean_kl": mean_kl,
                                "pearson_entropy_vs_ks": r_p_ks,
                                "spearman_entropy_vs_ks": r_s_ks,
                                "pearson_entropy_hat_vs_real": r_p_real,
                                "err": "",
                            }]).to_csv(out, mode="a", index=False,
                                       header=not header_written)
                            header_written = True
                            rows_written += 1
                            if r_s_ks > best_sp:
                                best_sp = r_s_ks
                                best_temp = temperature
                        print(f"  {tag}: best_τ={best_temp} "
                              f"spearman={best_sp:+.3f}")
                    except Exception as e:
                        msg = str(e)[:200]
                        print(f"  [err] {tag}: {msg}")
                        pd.DataFrame([{
                            "seed": seed, "T": T, "N": N, "gamma": gamma,
                            "temperature": float("nan"),
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
