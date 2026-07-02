"""Phase B: does the k=4 kernel improvement move the ORC sign signal?

Per (dataset in {plane, sphere, saddle}, seed in {7,8,9,10}) trains four nets
— measures role (gamma=0.8) x k in {1,4} and distances role (gamma=0.98) x
k in {1,4}, gamma_k = gamma^k — and evaluates the two-net ORC for variants:

  baseline : meas k=1, dist k=1   (= the measure-ablation relu baseline)
  meas_k4  : meas k=4, dist k=1   (Phase-A fit gains applied to measures)
  dist_k4  : meas k=1, dist k=4
  both_k4  : meas k=4, dist k=4

Clipped kernels are saved (float16) for the ensemble post-pass, which
averages the 4 seeds' kernels *before* OT:

  pixi run python expressiveness_phaseB.py run --worker-id K --num-workers 2 --device cuda:K
  pixi run python expressiveness_phaseB.py ensemble
  pixi run python expressiveness_phaseB.py summarize

Gate B: plane-saddle separation >= 2 s.d. of seed noise, sphere-plane intact.
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

from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from expressiveness_gate import strided_trajectories
from fb_kernel_measure_ablation import build_X, raw_scores
from fb_kernel_orc_sanity import potential_distances, remetrize

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
N_TRAJ, TRAJ_LEN = 500, 50
Z_DIM = 64
N_ANCHORS = 30
SEEDS = (7, 8, 9, 10)
DATASETS = ("plane", "sphere", "saddle")
FB_BASE = dict(hidden_dim=256, n_epochs=600, cosine_lr=True, batch_size=1024)

ROLES = {  # role -> (gamma, ks)
    "meas": (0.8, (1, 4)),
    "dist": (0.98, (1, 4)),
}
VARIANTS = {  # variant -> (meas_k, dist_k)
    "baseline": (1, 1),
    "meas_k4": (4, 1),
    "dist_k4": (1, 4),
    "both_k4": (4, 4),
}

KDIR = Path("processed_data/phaseB_kernels")
OUT_TPL = "processed_data/expressiveness_phaseB_w{wid}.csv"
OUT_MERGED = Path("processed_data/expressiveness_phaseB.csv")


def rownorm_relu(K: np.ndarray) -> np.ndarray:
    K = np.maximum(K, 0.0) + 1e-12
    return K / K.sum(axis=1, keepdims=True)


def orc_eval(meas: np.ndarray, D: np.ndarray, anchors) -> tuple[float, float, int]:
    est = WassersteinSignedCurvature(n_pairs=8, seed=0, compute_midpoint=False)
    est.fit(M=meas, D=D, idx=anchors)
    return (float(np.nanmean(est.orc_)), float(np.nanstd(est.orc_)),
            int(np.isfinite(est.orc_).sum()))


def kernels_for(ds: str, seed: int, device: str) -> dict[str, np.ndarray]:
    """Train (or load) the four clipped kernels for one (dataset, seed)."""
    KDIR.mkdir(parents=True, exist_ok=True)
    path = KDIR / f"{ds}_s{seed}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k].astype(np.float64) for k in z.files}

    X, _ = build_X(ds, seed=seed)
    X32 = np.asarray(X, dtype=np.float32)
    G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
    traj = subsample_trajectories(G, n_trajectories=N_TRAJ, length=TRAJ_LEN,
                                  rng=seed)
    out = {}
    for role, (gamma, ks) in ROLES.items():
        for k in ks:
            tr = FBTrainer(obs_dim=X32.shape[1], z_dim=Z_DIM, gamma=gamma**k,
                           **FB_BASE, device=device, seed=seed)
            tr.fit(X32[strided_trajectories(traj, k)])
            K = np.maximum(raw_scores(tr, X32, device), 0.0)
            out[f"{role}_k{k}"] = K
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in out.items()})
    return out


def run_worker(args) -> None:
    units = list(itertools.product(DATASETS, SEEDS))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        done = set(map(tuple, pd.read_csv(out)[["dataset", "seed"]].values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units on {args.device}", flush=True)

    t0 = time.time()
    for prog, (ds, seed) in enumerate(mine, 1):
        if (ds, seed) in done:
            continue
        t1 = time.time()
        kern = kernels_for(ds, seed, args.device)
        anchors = np.random.default_rng(seed).choice(
            N_POINTS, N_ANCHORS, replace=False).tolist()
        Ds = {k: remetrize(potential_distances(kern[f"dist_k{k}"]))
              for k in ROLES["dist"][1]}
        rows = []
        for variant, (mk, dk) in VARIANTS.items():
            o, s, n_ok = orc_eval(rownorm_relu(kern[f"meas_k{mk}"]),
                                  Ds[dk], anchors)
            rows.append(dict(dataset=ds, seed=seed, variant=variant,
                             orc=o, orc_std=s, n_ok=n_ok))
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not header)
        header = True
        summary = " ".join(f"{r['variant']}={r['orc']:+.3f}" for r in rows)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} {ds} s={seed} "
              f"({time.time()-t1:.0f}s): {summary}", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_ensemble(args) -> None:
    """Average the seeds' clipped kernels before OT, per variant."""
    rows = []
    for ds in DATASETS:
        kerns = []
        for seed in SEEDS:
            path = KDIR / f"{ds}_s{seed}.npz"
            with np.load(path) as z:
                kerns.append({k: z[k].astype(np.float64) for k in z.files})
        anchors = np.random.default_rng(SEEDS[0]).choice(
            N_POINTS, N_ANCHORS, replace=False).tolist()
        for variant, (mk, dk) in VARIANTS.items():
            K_meas = np.mean([kk[f"meas_k{mk}"] for kk in kerns], axis=0)
            K_dist = np.mean([kk[f"dist_k{dk}"] for kk in kerns], axis=0)
            D = remetrize(potential_distances(K_dist))
            o, s, n_ok = orc_eval(rownorm_relu(K_meas), D, anchors)
            rows.append(dict(dataset=ds, seed=-1, variant=f"ens4_{variant}",
                             orc=o, orc_std=s, n_ok=n_ok))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("processed_data/expressiveness_phaseB_ens.csv",
                              index=False)


def run_summarize(args) -> None:
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("expressiveness_phaseB_w*.csv"))]
    ens = Path("processed_data/expressiveness_phaseB_ens.csv")
    if ens.exists():
        frames.append(pd.read_csv(ens))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "seed", "variant"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    seeded = df[df.seed >= 0]
    print("orc mean ± std over seeds:")
    print(seeded.groupby(["variant", "dataset"]).orc.agg(["mean", "std"])
          .round(4).to_string())
    piv = seeded.pivot_table(index=["variant", "seed"], columns="dataset",
                             values="orc")
    piv["sphere-plane"] = piv.sphere - piv.plane
    piv["plane-saddle"] = piv.plane - piv.saddle
    agg = piv.groupby("variant")[["sphere-plane", "plane-saddle"]].agg(
        ["mean", "std"])
    print("\nGate B separations (want plane-saddle mean >= 2*std):")
    print(agg.round(4).to_string())
    if ens.exists():
        print("\nensemble-averaged kernels (single estimate, 30 anchors):")
        e = df[df.seed < 0].pivot_table(index="variant", columns="dataset",
                                        values="orc")
        e["sphere-plane"] = e.sphere - e.plane
        e["plane-saddle"] = e.plane - e.saddle
        print(e.round(4).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("ensemble")
    sub.add_parser("summarize")
    args = p.parse_args()
    {"run": run_worker, "ensemble": run_ensemble,
     "summarize": run_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
