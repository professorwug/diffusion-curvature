"""T1 — trajectory-regime channel survival curves.

Per (colosseum instance m<15, n_traj in {10,25,50,100,250}): sample walks
(T=50) on the instance's kNN graph; estimators see ONLY the visited unique
coordinates X_V. Channels (euclidean visited-cloud versions):

  kappa_plus : Diffusion ORC t=8 at the visited-node-nearest-origin
  frac       : diffusing-edge fraction, origin-local 5x16 edges, t=16
  ent_cak_t{2,4,8} : entropy ruler on the curvature-agnostic kernel with
                     coverage-scaled target N = max(|V|/4, 20)

Records coverage, |V|, eval_dist. Summarize: Pearson + flat-zero balanced
sign per (channel, n_traj, dim) — the survival map that allocates T2/T3.

Usage:
  pixi run python traj_t1_survival.py run --worker-id K --num-workers W --device cuda:X
  pixi run python traj_t1_survival.py summarize
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
import torch

from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from benchmark_kmetric_colosseum import affinity_from_D
from diffusing_fraction_colosseum import edge_fractions
from graph_ablation_ruler import cak_W

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
NTS = (10, 25, 50, 100, 250)
TRAJ_LEN = 50
KNN = 10
T_ORC = 8
T_FRAC = 16
K_EDGE = 16
N_LOCAL = 5
TS_CAK = (2, 4, 8)
MIN_V = 60

OUT_TPL = "processed_data/traj_t1_w{wid}.csv"
OUT_MERGED = Path("processed_data/traj_t1.csv")

CH_KEYS = ("kappa_plus", "frac", *[f"ent_cak_t{t}" for t in TS_CAK])


def traj_channels(X_V: np.ndarray, i0: int, device: str) -> dict[str, float]:
    out: dict[str, float] = {}
    nV = X_V.shape[0]
    k_eff = min(KNN, nV - 2)

    est = WassersteinSignedCurvature(t=T_ORC, knn=k_eff, n_pairs=8, seed=0,
                                     compute_midpoint=False)
    est.fit(X=X_V, idx=[i0])
    out["kappa_plus"] = float(est.orc_[0])

    Xt = torch.as_tensor(X_V, dtype=torch.float32, device=device)
    D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
    order0 = np.argsort(D[i0])
    anchors = [i0] + [int(j) for j in order0[1:N_LOCAL]]

    W10 = affinity_from_D(D, k=k_eff)
    P10 = torch.as_tensor(
        W10 / np.maximum(W10.sum(axis=1, keepdims=True), 1e-30),
        dtype=torch.float32, device=device)
    order = np.argsort(D[anchors], axis=1)
    k_edge = min(K_EDGE, nV - 2)
    edges = [(a, int(j)) for q, a in enumerate(anchors)
             for j in order[q, 1:k_edge + 1]]
    fr = edge_fractions(P10, edges, (T_FRAC,))
    out["frac"] = float(fr[T_FRAC].mean())

    W_cak, _ = cak_W(D, max(nV // 4, 20))
    P_cak = torch.as_tensor(
        W_cak / np.maximum(W_cak.sum(axis=1, keepdims=True), 1e-30),
        dtype=torch.float32, device=device)
    with torch.no_grad():
        rows = torch.zeros((len(anchors), nV), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for step in range(1, max(TS_CAK) + 1):
            rows = rows @ P_cak
            if step in TS_CAK:
                pr = rows.clamp_min(1e-12)
                out[f"ent_cak_t{step}"] = float(
                    (-(pr * pr.log()).sum(dim=1)).mean())
    return out


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
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)

    t0 = time.time()
    for prog, (i, inst, nt) in enumerate(mine, 1):
        if (i, nt) in done:
            continue
        X = np.asarray(inst["X"], dtype=np.float64)
        err = ""
        ch: dict[str, float] = {k: np.nan for k in CH_KEYS}
        coverage = eval_dist = np.nan
        nV = 0
        try:
            G = pygsp.graphs.NNGraph(X, k=KNN)
            traj_idx = subsample_trajectories(
                G, n_trajectories=nt, length=TRAJ_LEN, rng=1000 + nt)
            V = np.unique(traj_idx)
            nV = len(V)
            coverage = nV / X.shape[0]
            X_V = X[V]
            d0 = np.linalg.norm(X_V - X[0], axis=1)
            i0 = int(np.argmin(d0))
            eval_dist = float(d0[i0])
            if nV >= MIN_V:
                ch = traj_channels(X_V, i0, args.device)
        except Exception as e:
            err = str(e)[:200]
            print(f"  [err] inst={i} nt={nt}: {err}", flush=True)
        pd.DataFrame([dict(
            instance=i, n_traj=nt, err=err, name=inst["name"],
            dim=inst["dim"], noise=inst["noise"], ks_true=inst["ks_true"],
            coverage=round(coverage, 3) if np.isfinite(coverage) else np.nan,
            n_visited=nV, eval_dist=round(eval_dist, 4)
            if np.isfinite(eval_dist) else np.nan, **ch)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
        if prog % 20 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("traj_t1_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance", "n_traj"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)")
    print("\ncoverage | eval_dist medians per n_traj:")
    print(df.groupby("n_traj")[["coverage", "eval_dist", "n_visited"]]
          .median().round(3).to_string())

    # channel orientation: higher = more positive K
    for ch, orient in [("kappa_plus", +1), ("frac", -1),
                       *[(f"ent_cak_t{t}", -1) for t in TS_CAK]]:
        print(f"\n=== {ch}: Pearson | flat-zero balanced sign per (n_traj, dim) ===")
        rows = []
        for nt in NTS:
            row = {"n_traj": nt}
            for d in (2, 3, 4, 5, 6):
                g = df[(df.n_traj == nt) & (df.dim == d)]
                v = orient * pd.to_numeric(g[ch], errors="coerce")
                m = np.isfinite(v)
                if m.sum() > 5 and v[m].std() > 0:
                    r = pearsonr(v[m], g.ks_true[m])[0]
                    thr = g[m][g[m].ks_true.abs()
                              <= g[m].ks_true.abs().quantile(0.33)]
                    zero = orient * pd.to_numeric(
                        thr[ch], errors="coerce").median()
                    pos = m & (g.ks_true > 0)
                    neg = m & (g.ks_true < 0)
                    bal = (0.5 * ((v[pos] > zero).mean()
                                  + (v[neg] < zero).mean())
                           if pos.any() and neg.any() else np.nan)
                    row[f"d{d}"] = f"{r:+.2f}|{bal:.2f}"
                else:
                    row[f"d{d}"] = "  -  "
            rows.append(row)
        print(pd.DataFrame(rows).set_index("n_traj").to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        run_worker(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()
