"""Measure-construction ablation for FB-Kernel ORC: does un-clipping the
kernel scores restore the negative-curvature (saddle) signal?

Hypothesis (Stage-2 uncertainty #1): kappa < 0 requires accurate measure
tails — mass spreading into the saddle wings — and the ReLU clip zeroes
exactly those low-score entries. Test tail-preserving constructions of the
gamma=0.8 measures from RAW (unclipped) FB scores:

  relu           : max(K, 0), rownorm            [Stage-2 baseline]
  softplus_sharp : softplus(K / (0.25 std)), rownorm
  softplus_wide  : softplus(K / std), rownorm
  shift          : K - min(K), rownorm           [extreme tail retention]

Distances unchanged (gamma=0.98 kernel, potential + Dijkstra re-metrization,
as validated). Seed-ensembled (FB training is nondeterministic even at fixed
seed). kappa averaged over 30 anchors x 8 pairs.

Usage:
  pixi run python fb_kernel_measure_ablation.py --worker-id 0 --num-workers 2 --device cuda:0
  pixi run python fb_kernel_measure_ablation.py --summarize
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
import torch

from diffusion_curvature.datasets import plane, rejection_sample_from_saddle, sphere
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from fb_kernel_orc_sanity import potential_distances, remetrize

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
N_TRAJ, TRAJ_LEN = 500, 50
Z_DIM = 64
GAMMA_DIST, GAMMA_MEAS = 0.98, 0.8
N_ANCHORS = 30
SEEDS = (7, 8, 9, 10)
DATASETS = ("saddle", "plane", "sphere")
FB_KW = dict(hidden_dim=256, n_epochs=600, cosine_lr=True, batch_size=1024)

OUT_TPL = "processed_data/fb_measure_ablation_w{wid}.csv"
OUT_MERGED = Path("processed_data/fb_measure_ablation.csv")


def raw_scores(trainer: FBTrainer, X: np.ndarray, device: str) -> np.ndarray:
    """Unclipped symmetric kernel scores min(F1 B^T, F2 B^T), symmetrized."""
    trainer.F_net.eval()
    trainer.B_net.eval()
    with torch.no_grad():
        o = torch.as_tensor(X, dtype=torch.float32, device=device)
        F1, F2 = trainer.F_net(o)
        Bc = trainer.B_net(o)
        K = torch.minimum(F1 @ Bc.T, F2 @ Bc.T).cpu().numpy()
    return 0.5 * (K + K.T)


def measure_variants(K_raw: np.ndarray) -> dict[str, np.ndarray]:
    s = float(np.std(K_raw))

    def rownorm(A):
        A = A + 1e-12
        return A / A.sum(axis=1, keepdims=True)

    def softplus(x):
        return np.logaddexp(0.0, x)

    return {
        "relu": rownorm(np.maximum(K_raw, 0.0)),
        "softplus_sharp": rownorm(softplus(K_raw / (0.25 * s))),
        "softplus_wide": rownorm(softplus(K_raw / s)),
        "shift": rownorm(K_raw - K_raw.min()),
    }


def build_X(name: str, seed: int) -> tuple[np.ndarray, float]:
    if name == "plane":
        X = np.hstack([plane(N_POINTS, dim=2), np.zeros((N_POINTS, 1))])
        return X, 0.0
    if name == "sphere":
        X, _ = sphere(N_POINTS, d=2, seed=seed)
        return np.asarray(X), 2.0
    Xs, ks = rejection_sample_from_saddle(N_POINTS, 2)
    return np.asarray(Xs), float(ks)


def run(args) -> None:
    units = list(itertools.product(DATASETS, SEEDS))
    mine = [u for k, u in enumerate(units) if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out)
        done = set(zip(prev.dataset, prev.seed))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units on {args.device}", flush=True)

    t0 = time.time()
    for prog, (ds, seed) in enumerate(mine, 1):
        if (ds, seed) in done:
            continue
        rng = np.random.default_rng(seed)
        X, ks = build_X(ds, seed)
        X32 = np.asarray(X, dtype=np.float32)
        anchors = rng.choice(N_POINTS, N_ANCHORS, replace=False).tolist()

        G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
        traj_idx = subsample_trajectories(
            G, n_trajectories=N_TRAJ, length=TRAJ_LEN, rng=seed)
        traj = X32[traj_idx]

        nets = {}
        for gamma in (GAMMA_DIST, GAMMA_MEAS):
            tr = FBTrainer(obs_dim=X.shape[1], z_dim=Z_DIM, gamma=gamma,
                           **FB_KW, device=args.device, seed=seed)
            tr.fit(traj)
            nets[gamma] = tr

        K_dist = np.maximum(raw_scores(nets[GAMMA_DIST], X32, args.device), 0.0)
        D = remetrize(potential_distances(K_dist))
        K_meas_raw = raw_scores(nets[GAMMA_MEAS], X32, args.device)

        rows = []
        for variant, meas in measure_variants(K_meas_raw).items():
            est = WassersteinSignedCurvature(n_pairs=8, seed=0,
                                             compute_midpoint=False)
            est.fit(M=meas, D=D, idx=anchors)
            rows.append(dict(
                dataset=ds, seed=seed, variant=variant, ks_true=ks,
                orc=float(np.nanmean(est.orc_)),
                orc_std=float(np.nanstd(est.orc_)),
                n_ok=int(np.isfinite(est.orc_).sum()),
                spread=float(np.nanmean(est.spread_)),
            ))
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not header)
        header = True
        summary = " ".join(f"{r['variant']}={r['orc']:+.3f}" for r in rows)
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} {ds} seed={seed}: "
              f"{summary} eta={(len(mine)-prog)/max(rate,1e-9):.0f}min",
              flush=True)

    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def summarize() -> None:
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("fb_measure_ablation_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "seed", "variant"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    g = df.groupby(["variant", "dataset"]).orc.agg(["mean", "std", "count"])
    print("kappa mean ± std over seeds (30 anchors each):")
    print(g.round(4).to_string())
    print("\nSeparations (mean over seeds):")
    piv = df.pivot_table(index=["variant", "seed"], columns="dataset", values="orc")
    piv["sphere-plane"] = piv.sphere - piv.plane
    piv["plane-saddle"] = piv.plane - piv.saddle
    agg = piv.groupby("variant")[["sphere-plane", "plane-saddle"]].agg(["mean", "std"])
    print(agg.round(4).to_string())
    print("\nper-seed saddle sign (negative = correct):")
    print(df[df.dataset == "saddle"].pivot_table(
        index="variant", columns="seed", values="orc").round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args)


if __name__ == "__main__":
    main()
