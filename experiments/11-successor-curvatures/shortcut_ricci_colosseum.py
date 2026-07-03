"""Shortcut-value Ricci on the Colosseum battery (trajectory regime).

The v4 winner (`shortcut_odd_q60`): odd entropy response to a virtual
shortcut planted at the 55-65th geodesic-distance quantile, on the visited
cloud's euclidean affinity graph. Training-free. Signed convention: HIGHER
response = MORE POSITIVE curvature (established on the v4 triads at d>=4;
d=2 known inverted).

Per (colosseum instance m<15, n_traj in {100, 500}): score at 20 anchors near
the origin-side of the visited cloud... anchors = visited node nearest the
origin + 19 random (origin anchor recorded separately). Rows: score_origin,
score_mean.

Usage:
  pixi run python shortcut_ricci_colosseum.py run --worker-id K --num-workers 4 --device cuda:X
  pixi run python shortcut_ricci_colosseum.py summarize
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pygsp
from sklearn.metrics import pairwise_distances

from diffusion_curvature.trajectory_utils import subsample_trajectories

from benchmark_kmetric_colosseum import affinity_from_D
from successor_ricci_ladder import (
    _auto_t_dense,
    fd_edge_response,
    quantile_band_edges,
)

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
OUT_TPL = "processed_data/shortcut_ricci_w{wid}.csv"
OUT_MERGED = Path("processed_data/shortcut_ricci.csv")

M_MAX = 15
NTS = (100, 500)
TRAJ_LEN = 50
KNN = 10
N_ANCHORS = 20
K_EDGE = 4
Q_LO, Q_HI = 0.55, 0.65


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["m"] < M_MAX]
    units = [(i, inst, nt) for (i, inst), nt in itertools.product(cc, NTS)]
    mine = [u for k, u in enumerate(units) if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["instance", "n_traj"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units on {args.device}", flush=True)

    t0 = time.time()
    for prog, (i, inst, nt) in enumerate(mine, 1):
        if (i, nt) in done:
            continue
        t1 = time.time()
        X = np.asarray(inst["X"], dtype=np.float64)
        rng = np.random.default_rng(7)
        err = ""
        try:
            G = pygsp.graphs.NNGraph(X, k=KNN)
            traj_idx = subsample_trajectories(G, n_trajectories=nt,
                                              length=TRAJ_LEN, rng=1000 + nt)
            V = np.unique(traj_idx)
            X_V = X[V]
            i0 = int(np.argmin(np.linalg.norm(X_V - X[0], axis=1)))
            anchors = [i0] + rng.choice(
                len(V), min(N_ANCHORS - 1, len(V) - 1), replace=False).tolist()

            D_euc = pairwise_distances(X_V)
            W_euc = affinity_from_D(D_euc, k=KNN)
            t_euc = _auto_t_dense(W_euc, D_euc, anchors)
            edges = quantile_band_edges(D_euc, anchors, Q_LO, Q_HI,
                                        K_EDGE, rng)
            odd, _ = fd_edge_response(W_euc, t_euc, edges, args.device)
            n_a = len(odd) // K_EDGE
            by_anchor = odd[:n_a * K_EDGE].reshape(n_a, K_EDGE).mean(axis=1)
            score_origin = float(by_anchor[0])
            score_mean = float(np.mean(by_anchor))
            coverage = len(V) / X.shape[0]
        except Exception as e:
            score_origin = score_mean = float("nan")
            coverage = float("nan")
            err = str(e)[:200]
            print(f"  [err] inst={i}: {err}", flush=True)

        pd.DataFrame([dict(
            instance=i, n_traj=nt, method="shortcut_odd_q60",
            score_origin=score_origin, score_mean=score_mean,
            coverage=round(coverage, 3) if np.isfinite(coverage) else "",
            err=err, name=inst["name"], dim=inst["dim"], noise=inst["noise"],
            m=inst["m"], ks_true=inst["ks_true"],
        )]).to_csv(out, mode="a", index=False, header=not header)
        header = True
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} inst={i} "
              f"d={inst['dim']} nt={nt} ({time.time()-t1:.0f}s) "
              f"score={score_mean:+.3g} eta={(len(mine)-prog)/max(rate,1e-9):.0f}min",
              flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr, spearmanr
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("shortcut_ricci_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["instance", "n_traj"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    for col in ("score_mean", "score_origin"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ok = df[df.score_mean.notna()]
    print(f"wrote {OUT_MERGED} ({len(df)} rows, {len(ok)} finite)\n")
    for col in ("score_mean", "score_origin"):
        rows = []
        for (nt, d), g in ok.groupby(["n_traj", "dim"]):
            if len(g) > 3 and g[col].std() > 0:
                pos, neg = g[g.ks_true > 0], g[g.ks_true < 0]
                bal = (0.5 * ((pos[col] > pos[col].median()).mean()
                              + 1 - (neg[col] > neg[col].median()).mean())
                       if len(pos) and len(neg) else np.nan)
                rows.append(dict(
                    n_traj=nt, dim=d,
                    pearson=pearsonr(g[col], g.ks_true)[0],
                    spearman=spearmanr(g[col], g.ks_true)[0],
                    auc_like=np.nan))
        s = pd.DataFrame(rows)
        print(f"=== {col}: correlation with ks_true ===")
        print(s.pivot_table(index="n_traj", columns="dim",
                            values=["pearson", "spearman"]).round(2).to_string())
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
