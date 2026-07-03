"""Higher-dimensional ablation ladder for successor-based signed ORC.

Motivated by the user's observation that successor measures are "oddly
pathological" in 2D (candidate mechanism: Pólya recurrence — walks are
recurrent in 2D, transient in d>=3, so the high-gamma occupancy ratio is
near-singular only in 2D). Repeats the key design tests at d=4 alongside d=2:

Per (dim in {2,4}, dataset in {plane, sphere, saddle}), one unit trains
(vmapped, 4 seeds, 300 epochs):
  - distance ensemble  (gamma=0.98)          -> potential distances, D_graph
  - measure ensembles  (gamma in {0.5, 0.8, 0.95})

and evaluates variants:
  relu_g{0.5,0.8,0.95} : two-net ORC, rownorm(relu(K_meas)) measures
  softplus_g0.8        : tail-preserving measures (did the d=2 verdict flip?)
  kmetric              : diffusion measures on the learned metric (t='auto')

kappa at 30 anchors x 8 pairs, per seed; CSV row per (unit, variant, seed).

Usage:
  pixi run python fb_kernel_ablation_d4.py run --worker-id K --num-workers 2 --device cuda:0
  pixi run python fb_kernel_ablation_d4.py summarize
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

from diffusion_curvature.datasets import plane, rejection_sample_from_saddle, sphere
from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from benchmark_kmetric_colosseum import (
    _SimpleG,
    affinity_from_D,
    knn_distance_graph,
    potential_distances,
)

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
N_TRAJ, TRAJ_LEN = 500, 50
Z_DIM = 64
N_EPOCHS = 300
SEEDS = (7, 8, 9, 10)
N_ANCHORS = 30
DIMS = (2, 4)
DATASETS = ("plane", "sphere", "saddle")
GAMMA_DIST = 0.98
GAMMAS_MEAS = (0.5, 0.8, 0.95)

OUT_TPL = "processed_data/fb_ablation_d4_w{wid}.csv"
OUT_MERGED = Path("processed_data/fb_ablation_d4.csv")


def build_Xd(name: str, d: int, seed: int):
    if name == "plane":
        X = np.hstack([plane(N_POINTS, dim=d), np.zeros((N_POINTS, 1))])
        return np.asarray(X), 0.0
    if name == "sphere":
        X, _ = sphere(N_POINTS, d=d, seed=seed)
        return np.asarray(X), float(d * (d - 1))
    X, ks = rejection_sample_from_saddle(N_POINTS, d)
    return np.asarray(X), float(ks)


def softplus_rownorm(K_raw: np.ndarray) -> np.ndarray:
    s = float(np.std(K_raw))
    M = np.logaddexp(0.0, K_raw / (0.25 * s)) + 1e-12
    return M / M.sum(axis=1, keepdims=True)


def relu_rownorm(K_raw: np.ndarray) -> np.ndarray:
    M = np.maximum(K_raw, 0.0) + 1e-12
    return M / M.sum(axis=1, keepdims=True)


def run_worker(args) -> None:
    units = list(itertools.product(DIMS, DATASETS))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["dim", "dataset", "variant"])
        done = set(zip(prev.dim, prev.dataset, prev.variant))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units on {args.device}", flush=True)

    t0 = time.time()
    for prog, (d, ds) in enumerate(mine, 1):
        t1 = time.time()
        X, ks = build_Xd(ds, d, seed=7)
        X32 = X.astype(np.float32)
        rng = np.random.default_rng(7)
        anchors = rng.choice(N_POINTS, N_ANCHORS, replace=False).tolist()
        G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
        traj = X32[subsample_trajectories(G, n_trajectories=N_TRAJ,
                                          length=TRAJ_LEN, rng=7)]

        ens_d = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                                  gamma=GAMMA_DIST, n_epochs=N_EPOCHS,
                                  device=args.device)
        ens_d.fit(traj)
        K_dist = np.maximum(ens_d.raw_kernels(X32), 0.0)
        D_pots = [potential_distances(K_dist[s]) for s in range(len(SEEDS))]
        Gks = [knn_distance_graph(Dp) for Dp in D_pots]

        meas_by_gamma = {}
        for gm in GAMMAS_MEAS:
            ens_m = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS,
                                      z_dim=Z_DIM, gamma=gm,
                                      n_epochs=N_EPOCHS, device=args.device)
            ens_m.fit(traj)
            meas_by_gamma[gm] = ens_m.raw_kernels(X32)  # raw (unclipped)

        variants: dict[str, list[float]] = {}
        for s in range(len(SEEDS)):
            for gm in GAMMAS_MEAS:
                name = f"relu_g{gm:g}"
                try:
                    est = WassersteinSignedCurvature(
                        n_pairs=8, seed=0, compute_midpoint=False)
                    est.fit(M=relu_rownorm(meas_by_gamma[gm][s]),
                            D_graph=Gks[s], idx=anchors)
                    variants.setdefault(name, []).append(
                        float(np.nanmean(est.orc_)))
                except Exception:
                    pass
            try:
                est = WassersteinSignedCurvature(
                    n_pairs=8, seed=0, compute_midpoint=False)
                est.fit(M=softplus_rownorm(meas_by_gamma[0.8][s]),
                        D_graph=Gks[s], idx=anchors)
                variants.setdefault("softplus_g0.8", []).append(
                    float(np.nanmean(est.orc_)))
            except Exception:
                pass
            try:
                W = affinity_from_D(D_pots[s])
                est = WassersteinSignedCurvature(
                    t="auto", n_pairs=8, seed=0, compute_midpoint=False)
                est.fit(G=_SimpleG(W), D_graph=Gks[s], idx=anchors)
                variants.setdefault("kmetric", []).append(
                    float(np.nanmean(est.orc_)))
            except Exception:
                pass

        rows = []
        for name, vals in variants.items():
            if (d, ds, name) in done:
                continue
            for s_i, v in enumerate(vals):
                rows.append(dict(dim=d, dataset=ds, variant=name,
                                 seed=SEEDS[s_i] if s_i < len(SEEDS) else -1,
                                 ks_true=ks, orc=v))
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", index=False,
                                      header=not header)
            header = True
        means = {n: np.mean(v) for n, v in variants.items()}
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} d={d} {ds} "
              f"({time.time()-t1:.0f}s): "
              + " ".join(f"{n}={m:+.3f}" for n, m in means.items()), flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize(args) -> None:
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("fb_ablation_d4_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dim", "dataset", "variant", "seed"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    g = df.groupby(["dim", "variant", "dataset"]).orc.agg(["mean", "std"])
    print(g.round(4).to_string())
    piv = df.pivot_table(index=["dim", "variant", "seed"], columns="dataset",
                         values="orc")
    piv["sphere-plane"] = piv.sphere - piv.plane
    piv["plane-saddle"] = piv.plane - piv.saddle
    agg = piv.groupby(["dim", "variant"])[
        ["sphere-plane", "plane-saddle"]].agg(["mean", "std"])
    print("\nseparations (want both positive, >= 2x std):")
    print(agg.round(4).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = p.parse_args()
    if args.cmd == "run":
        run_worker(args)
    else:
        run_summarize(args)


if __name__ == "__main__":
    main()
