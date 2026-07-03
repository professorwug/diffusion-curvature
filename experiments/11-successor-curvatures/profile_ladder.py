"""Validation ladder for the kappa(d) profile estimator.

kappa(d) = c0 + c1/d + c2*d^2 fitted over pair-distance bands; c2 (the d^2
slope) is the contamination-free signed curvature readout. Pass criteria per
(dim, n_traj, arm):
  plane  : standardized slope ~ 0
  sphere : slope > 0  (and NOT penalized, unlike the single-far-band cal)
  saddle : slope < 0
Arms: learned (FB measures gamma=0.8 + learned metric) and unlearned control
(visited cloud, auto-t). Slopes are pooled across anchors per seed
(homogeneous triad manifolds), standardized as c2 * median(d)^2.

Usage: pixi run python profile_ladder.py --worker-id K --num-workers 2 --device cuda:0
       pixi run python profile_ladder.py --summarize
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from benchmark_kmetric_colosseum import knn_distance_graph, potential_distances
from fb_kernel_ablation_d4 import build_Xd, relu_rownorm

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
TRAJ_LEN = 50
Z_DIM = 64
N_EPOCHS = 300
SEEDS = (7, 8, 9, 10)
N_ANCHORS = 15
DIMS = (2, 4)
NTS = (50, 200)
DATASETS = ("plane", "sphere", "saddle")
BANDS = WassersteinSignedCurvature.DEFAULT_BANDS

OUT_TPL = "processed_data/profile_ladder_w{wid}.csv"
OUT_MERGED = Path("processed_data/profile_ladder.csv")


def pooled_fit(est) -> dict[str, float]:
    """Pool (d, kappa) pairs across anchors; fit the 3-term profile."""
    ds = np.concatenate([p[0] for p in est.profile_pairs_ if len(p[0])])
    ks = np.concatenate([p[1] for p in est.profile_pairs_ if len(p[1])])
    if len(ds) < 10:
        return dict(slope_std=np.nan, rho=np.nan, delta_term=np.nan,
                    raw_near=float(np.nanmean(est.orc_)), n_pairs=len(ds))
    A = np.stack([np.ones_like(ds), 1.0 / ds, ds**2], axis=1)
    coef, *_ = np.linalg.lstsq(A, ks, rcond=None)
    dmed = float(np.median(ds))
    return dict(
        slope_std=float(coef[2]) * dmed**2,   # kappa-like units
        rho=float(coef[0]),
        delta_term=-float(coef[1]) / dmed,    # kappa-like units at d_med
        raw_near=float(np.nanmean(est.orc_)),
        n_pairs=len(ds),
    )


def run_worker(args) -> None:
    units = list(itertools.product(DIMS, DATASETS, NTS))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["dim", "dataset", "n_traj", "arm"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units on {args.device}", flush=True)

    t0 = time.time()
    for prog, (d, ds_name, nt) in enumerate(mine, 1):
        t1 = time.time()
        X, ks_true = build_Xd(ds_name, d, seed=7)
        X32 = X.astype(np.float32)
        rng = np.random.default_rng(7)
        G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
        traj_idx = subsample_trajectories(G, n_trajectories=nt,
                                          length=TRAJ_LEN, rng=7)
        traj = X32[traj_idx]
        anchors = rng.choice(N_POINTS, N_ANCHORS, replace=False).tolist()

        rows = []

        # --- learned arm ---
        if (d, ds_name, nt, "learned") not in done:
            ens_d = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS,
                                      z_dim=Z_DIM, gamma=0.98,
                                      n_epochs=N_EPOCHS, device=args.device)
            ens_d.fit(traj)
            K_dist = np.maximum(ens_d.raw_kernels(X32), 0.0)
            ens_m = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS,
                                      z_dim=Z_DIM, gamma=0.8,
                                      n_epochs=N_EPOCHS, device=args.device)
            ens_m.fit(traj)
            K_meas = np.maximum(ens_m.raw_kernels(X32), 0.0)
            for s in range(len(SEEDS)):
                Gk = knn_distance_graph(potential_distances(K_dist[s]))
                est = WassersteinSignedCurvature(
                    n_pairs=8, seed=0, compute_midpoint=False,
                    profile_bands=BANDS)
                est.fit(M=relu_rownorm(K_meas[s]), D_graph=Gk, idx=anchors)
                rows.append(dict(dim=d, dataset=ds_name, n_traj=nt,
                                 arm="learned", seed=SEEDS[s],
                                 ks_true=ks_true, **pooled_fit(est)))

        # --- unlearned control arm ---
        if (d, ds_name, nt, "control") not in done:
            V = np.unique(traj_idx)
            anchor_pos = [int(np.argmin(np.linalg.norm(X[V] - X[a], axis=1)))
                          for a in anchors]
            est = WassersteinSignedCurvature(
                t="auto", knn=KNN, n_pairs=8, seed=0, compute_midpoint=False,
                profile_bands=BANDS)
            est.fit(X=X[V], idx=anchor_pos)
            rows.append(dict(dim=d, dataset=ds_name, n_traj=nt,
                             arm="control", seed=-1, ks_true=ks_true,
                             **pooled_fit(est)))

        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", index=False,
                                      header=not header)
            header = True
        by_arm = {}
        for r in rows:
            by_arm.setdefault(r["arm"], []).append(r["slope_std"])
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} d={d} {ds_name} nt={nt} "
              f"({time.time()-t1:.0f}s): "
              + " ".join(f"{a}_slope={np.nanmean(v):+.4f}"
                         for a, v in by_arm.items())
              + f" eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("profile_ladder_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dim", "dataset", "n_traj", "arm", "seed"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    for metric in ("slope_std", "raw_near", "rho", "delta_term"):
        print(f"=== {metric} (mean over seeds) ===")
        print(df.pivot_table(index=["arm", "dim", "n_traj"], columns="dataset",
                             values=metric).round(4).to_string())
        print()
    piv = df.pivot_table(index=["arm", "dim", "n_traj", "seed"],
                         columns="dataset", values="slope_std")
    piv["sphere-plane"] = piv.sphere - piv.plane
    piv["plane-saddle"] = piv.plane - piv.saddle
    print("=== slope separations (want both positive) ===")
    print(piv.groupby(["arm", "dim", "n_traj"])[
        ["sphere-plane", "plane-saddle"]].agg(["mean", "std"]).round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        run_summarize()
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
