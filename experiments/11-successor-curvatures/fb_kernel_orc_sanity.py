"""Stage 2: FB-Kernel ORC sign sanity on plane/sphere/saddle from trajectories.

Fully kernel-driven signed curvature: one FB net (gamma=0.98, z=64) trained on
random-walk trajectories; distances = PHATE-style potential transform of the
symmetrized kernel; measures = softmax(K/tau) rows with tau chosen so the
measure spread is ~0.3 of the median kernel-distance radius (two scales from
one net). No graph diffusions anywhere downstream of trajectory sampling.

Pass criteria: sphere > 0, saddle < 0, plane ~ 0 at the origin/anchors.

Usage: pixi run python fb_kernel_orc_sanity.py [--device cuda:0]
Writes processed_data/fb_kernel_orc_sanity.csv
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import pygsp
from sklearn.metrics import pairwise_distances

from diffusion_curvature.datasets import plane, rejection_sample_from_saddle, sphere
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
N_TRAJ, TRAJ_LEN = 500, 50
GAMMA, Z_DIM = 0.98, 64
SPREAD_FRACTION = 0.3
SEED = 7
FB_KW = dict(hidden_dim=256, n_epochs=600, cosine_lr=True, batch_size=1024)


def potential_distances(K: np.ndarray) -> np.ndarray:
    rs = K.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    U = -np.log((K / rs).astype(np.float64) + 1e-6)
    return pairwise_distances(U)


def remetrize(D: np.ndarray, k: int = KNN) -> np.ndarray:
    """Shortest-path metric over the k nearest raw distances per node.

    The -log potential transform saturates at range (concave in true
    geodesic distance), which biases W1/d ratios positive. Keeping only the
    trustworthy local distances and chaining them with Dijkstra restores
    additivity along paths.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    vals = D[rows, cols]
    Gk = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return dijkstra(Gk, directed=False)


def softmax_rows(K: np.ndarray, tau: float) -> np.ndarray:
    x = K / max(tau, 1e-12)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def pick_tau(K: np.ndarray, D: np.ndarray, rng: np.random.Generator) -> float:
    """Bisect tau (relative to K's scale) so the median measure spread hits
    SPREAD_FRACTION of the median kernel-distance radius."""
    probes = rng.choice(K.shape[0], size=20, replace=False)
    target = SPREAD_FRACTION * float(np.median(D[probes]))

    def spread(tau: float) -> float:
        P = softmax_rows(K[probes], tau)
        return float(np.median((P * D[probes]).sum(axis=1)))

    scale = float(np.std(K[probes]))
    lo, hi = 1e-4 * scale, 1e4 * scale  # sharp .. flat
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if spread(mid) > target:
            hi = mid  # too spread: sharpen
        else:
            lo = mid
    tau = np.sqrt(lo * hi)
    print(f"    pick_tau: scale={scale:.3g} tau={tau:.4g} "
          f"spread={spread(tau):.3f} target={target:.3f} "
          f"flat_spread={spread(hi * 100):.3f}", flush=True)
    return float(tau)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="processed_data/fb_kernel_orc_sanity.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    # 30 anchors each: sign is constant across all three manifolds (plane 0
    # everywhere; sphere uniform; saddle negative everywhere), so averaging
    # over anchors is valid for the sign test and beats single-point variance.
    anchors = rng.choice(N_POINTS, 30, replace=False).tolist()
    datasets = []
    Xp = np.hstack([plane(N_POINTS, dim=2), np.zeros((N_POINTS, 1))])
    datasets.append(("plane", Xp, 0.0, anchors))
    Xs, _ = sphere(N_POINTS, d=2, seed=SEED)
    datasets.append(("sphere", np.asarray(Xs), 2.0, anchors))
    Xsad, ksad = rejection_sample_from_saddle(N_POINTS, 2)
    datasets.append(("saddle", np.asarray(Xsad), float(ksad), anchors))

    rows = []
    for name, X, ks, idxs in datasets:
        t0 = time.time()
        G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
        traj_idx = subsample_trajectories(
            G, n_trajectories=N_TRAJ, length=TRAJ_LEN, rng=SEED)
        trainer = FBTrainer(obs_dim=X.shape[1], z_dim=Z_DIM, gamma=GAMMA,
                            **FB_KW, device=args.device, seed=SEED)
        trainer.fit(np.asarray(X, dtype=np.float32)[traj_idx])
        M, _, _ = compute_successor_measures(
            trainer.F_net, trainer.B_net,
            np.asarray(X, dtype=np.float32), np.asarray(X, dtype=np.float32),
            device=args.device)
        K = 0.5 * (M + M.T)
        D_raw = potential_distances(K)
        D_path = remetrize(D_raw)

        # Two-net two-scale: a moderate-gamma net supplies the *measures*
        # (true resolvent occupancies at a ~5-step horizon, no softmax
        # resharpening); the gamma=0.98 net supplies the *distances*.
        trainer_m = FBTrainer(obs_dim=X.shape[1], z_dim=Z_DIM, gamma=0.8,
                              **FB_KW, device=args.device, seed=SEED)
        trainer_m.fit(np.asarray(X, dtype=np.float32)[traj_idx])
        M_meas, _, _ = compute_successor_measures(
            trainer_m.F_net, trainer_m.B_net,
            np.asarray(X, dtype=np.float32), np.asarray(X, dtype=np.float32),
            device=args.device)
        K_meas = np.maximum(0.5 * (M_meas + M_meas.T), 0.0) + 1e-12
        meas_gamma08 = K_meas / K_meas.sum(axis=1, keepdims=True)

        variants = {
            "potential": (D_raw, None),
            "potential+dijkstra": (D_path, None),
            "twonet+dijkstra": (D_path, meas_gamma08),
        }

        for variant, (D, meas_pre) in variants.items():
            if meas_pre is None:
                tau = pick_tau(K, D, rng)
                meas = softmax_rows(K, tau)
            else:
                tau = np.nan
                meas = meas_pre
            est = WassersteinSignedCurvature(n_pairs=8, seed=0,
                                             compute_midpoint=False)
            est.fit(M=meas, D=D, idx=idxs)
            row = dict(
                dataset=name, variant=variant, ks_true=ks, tau=tau,
                orc=float(np.nanmean(est.orc_)),
                orc_std=float(np.nanstd(est.orc_)),
                orc_phys=float(np.nanmean(est.orc_phys_)),
                spread=float(np.nanmean(est.spread_)),
                elapsed_s=round(time.time() - t0, 1),
            )
            rows.append(row)
            print(f"{name:<7} [{variant:<18}] ks={ks:+6.2f} tau={tau:.4g} "
                  f"orc={row['orc']:+.4f}±{row['orc_std']:.4f} "
                  f"phys={row['orc_phys']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    for variant in df.variant.unique():
        o = {r["dataset"]: r["orc"] for r in rows if r["variant"] == variant}
        print(f"\n[{variant}] sphere>0: {o['sphere'] > 0}  "
              f"saddle<0: {o['saddle'] < 0}  |plane|: {abs(o['plane']):.4f}")


if __name__ == "__main__":
    main()
