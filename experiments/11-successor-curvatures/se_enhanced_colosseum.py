"""Enhanced Successor Entropy on the Colosseum battery (unsigned track).

SE stays in the one-distribution regime (no transport plans) but gets the
kernel-metric machinery:

  se_tau1        : H(softmax(K_0.8 / tau=1))            [baseline, fixed tau]
  se_tau_auto    : H(softmax(K_0.8 / tau_auto))         [spread-targeted tau]
  se_kmetric_t   : H(P^t rows of the kernel-distance graph, auto t)
                   [multi-step diffusion on the learned metric]
  w1_spread      : -sum_a mu(a) D_kernel(i,a)           [metric-aware spread]
  w2_spread      : -sqrt(sum_a mu(a) D_kernel(i,a)^2)
  se_multiscale  : H(mu_gamma=0.95) - H(mu_gamma=0.5)   [scale growth]

All scores are signed so that HIGHER should mean MORE POSITIVE curvature
(entropies/spreads negated). Per-instance scalar = mean over corpus nodes
(SE v2 convention). Metrics: Pearson/Spearman vs ks_true per (dim, noise).

Grid: signed battery colosseum, dims {2,3,5,6} x noise {0.01, 0.1} x m<15,
n_traj=500, 4 vmapped seeds.

Usage:
  pixi run python se_enhanced_colosseum.py run --worker-id K --num-workers 4 --device cuda:X
  pixi run python se_enhanced_colosseum.py summarize
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories

from benchmark_kmetric_colosseum import affinity_from_D, potential_distances
from fb_kernel_orc_sanity import softmax_rows
from successor_ricci_ladder import _auto_t_dense

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
OUT_TPL = "processed_data/se_enhanced_w{wid}.csv"
OUT_MERGED = Path("processed_data/se_enhanced.csv")

DIMS = (2, 3, 5, 6)
NOISES = (0.01, 0.1)
M_MAX = 15
N_TRAJ, TRAJ_LEN = 500, 50
KNN = 10
Z_DIM = 64
N_EPOCHS = 300
SEEDS = (7, 8)   # SE scores average over 3000 nodes; seed variance is low
GAMMAS = (0.5, 0.8, 0.95, 0.98)   # 0.98 = distances role
SPREAD_FRACTION = 0.3


def entropy_rows(P: np.ndarray) -> np.ndarray:
    P = np.maximum(P, 1e-15)
    P = P / P.sum(axis=1, keepdims=True)
    return -(P * np.log(P)).sum(axis=1)


def pick_tau_spread(K: np.ndarray, D: np.ndarray,
                    rng: np.random.Generator) -> float:
    probes = rng.choice(K.shape[0], size=20, replace=False)
    target = SPREAD_FRACTION * float(np.median(D[probes]))
    scale = float(np.std(K[probes]))
    lo, hi = 1e-4 * scale, 1e4 * scale
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        P = softmax_rows(K[probes], mid)
        if float(np.median((P * D[probes]).sum(axis=1))) > target:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def scores_for_instance(X: np.ndarray, device: str) -> dict[str, list[float]]:
    X32 = X.astype(np.float32)
    rng = np.random.default_rng(7)
    G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
    traj = X32[subsample_trajectories(G, n_trajectories=N_TRAJ,
                                      length=TRAJ_LEN, rng=7)]
    kernels = {}
    for gamma in GAMMAS:
        ens = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                                gamma=gamma, n_epochs=N_EPOCHS, device=device)
        ens.fit(traj)
        kernels[gamma] = np.maximum(ens.raw_kernels(X32), 0.0)

    out: dict[str, list[float]] = {}
    for s in range(len(SEEDS)):
        K08 = kernels[0.8][s]
        D_pot = potential_distances(kernels[0.98][s])

        out.setdefault("se_tau1", []).append(
            -float(entropy_rows(softmax_rows(K08, 1.0)).mean()))
        tau = pick_tau_spread(K08, D_pot, rng)
        mu_auto = softmax_rows(K08, tau)
        out.setdefault("se_tau_auto", []).append(
            -float(entropy_rows(mu_auto).mean()))

        W_k = affinity_from_D(D_pot, k=KNN)
        probes = rng.choice(X.shape[0], size=20, replace=False)
        t_k = _auto_t_dense(W_k, D_pot, probes)
        P = W_k / np.maximum(W_k.sum(axis=1, keepdims=True), 1e-30)
        Pt = np.linalg.matrix_power(P, t_k)
        out.setdefault("se_kmetric_t", []).append(
            -float(entropy_rows(Pt).mean()))

        mu = K08 + 1e-15
        mu = mu / mu.sum(axis=1, keepdims=True)
        out.setdefault("w1_spread", []).append(
            -float((mu * D_pot).sum(axis=1).mean()))
        out.setdefault("w2_spread", []).append(
            -float(np.sqrt((mu * D_pot**2).sum(axis=1)).mean()))

        def rn(K):
            K = K + 1e-15
            return K / K.sum(axis=1, keepdims=True)
        out.setdefault("se_multiscale", []).append(
            -float((entropy_rows(rn(kernels[0.95][s]))
                    - entropy_rows(rn(kernels[0.5][s]))).mean()))
    return out


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["dim"] in DIMS
          and inst["noise"] in NOISES and inst["m"] < M_MAX]
    mine = [u for k, u in enumerate(cc) if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        done = set(pd.read_csv(out, usecols=["instance"]).instance)
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} instances on {args.device}", flush=True)

    t0 = time.time()
    for prog, (i, inst) in enumerate(mine, 1):
        if i in done:
            continue
        t1 = time.time()
        try:
            sc = scores_for_instance(np.asarray(inst["X"], dtype=np.float64),
                                     args.device)
            err = ""
        except Exception as e:
            sc, err = {}, str(e)[:200]
            print(f"  [err] inst={i}: {err}", flush=True)
        rows = []
        for name, vals in sc.items():
            v = np.asarray(vals)
            v = v[np.isfinite(v)]
            rows.append(dict(instance=i, method=name,
                             score=float(v.mean()) if v.size else np.nan,
                             score_std=float(v.std()) if v.size else np.nan,
                             err=err, name=inst["name"], dim=inst["dim"],
                             noise=inst["noise"], m=inst["m"],
                             ks_true=inst["ks_true"]))
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", index=False,
                                      header=not header)
            header = True
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} inst={i} "
              f"d={inst['dim']} ({time.time()-t1:.0f}s) "
              f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr, spearmanr
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("se_enhanced_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["instance", "method"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    df = df[np.isfinite(df.score)]
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    for metric_name, fn in [("pearson", pearsonr), ("spearman", spearmanr)]:
        rows = []
        for (m, d, nz), g in df.groupby(["method", "dim", "noise"]):
            if len(g) > 3 and g.score.std() > 0:
                rows.append(dict(method=m, dim=d, noise=nz,
                                 r=fn(g.score, g.ks_true)[0]))
        s = pd.DataFrame(rows)
        print(f"=== {metric_name}(score, ks_true) ===")
        print(s.pivot_table(index="method", columns=["dim", "noise"],
                            values="r").round(2).to_string())
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=4)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        run_worker(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()
