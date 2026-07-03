"""Diffusing-edge fraction on the Colosseum battery (iid, training-free).

The estimator earned by diffusing_edge_probe.py: per anchor, the FRACTION of
probe kNN edges whose probability-space strengthening INCREASES walk entropy
(dH/deps > 0 at t past the spread rule). Scale-free by construction — the
first statistic in this program that needs no cross-manifold calibration.
Higher fraction = more negative curvature; we report NEGATED fractions so
higher = more positive (project convention).

Efficient response: P is never cloned; the edge perturbation is propagated
as rank-2 row corrections (rows @ P_e = rows @ P + rows[:,a] da + rows[:,j] dj).

Usage:
  pixi run python diffusing_fraction_colosseum.py run --worker-id K --num-workers 2 --device cuda:K
  pixi run python diffusing_fraction_colosseum.py summarize
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from benchmark_kmetric_colosseum import affinity_from_D

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
OUT_TPL = "processed_data/diffusing_fraction_w{wid}.csv"
OUT_MERGED = Path("processed_data/diffusing_fraction.csv")

M_MAX = 15
KNN = 10
K_EDGE = 8
N_ANCHORS = 30
TS = (8, 16)
EPS = 0.05


def edge_fractions(P: torch.Tensor, edges, ts, eps: float = EPS) -> dict[int, np.ndarray]:
    """Per-edge diffusing indicator (dH/deps > 0) at each t in ts.

    Rank-2 correction propagation; returns {t: bool array over edges}.
    """
    n = P.shape[0]
    device = P.device
    out = {t: [] for t in ts}
    t_max = max(ts)
    with torch.no_grad():
        for a, j in edges:
            da = eps * (-P[a].clone())
            da[j] += eps
            dj = eps * (-P[j].clone())
            dj[a] += eps
            rows0 = torch.zeros((2, n), device=device)
            rows0[0, a] = 1.0
            rows0[1, j] = 1.0
            rows_p, rows_0 = rows0.clone(), rows0.clone()
            for step in range(1, t_max + 1):
                rows_0 = rows_0 @ P
                rows_p = (rows_p @ P
                          + rows_p[:, a:a + 1] * da.unsqueeze(0)
                          + rows_p[:, j:j + 1] * dj.unsqueeze(0))
                if step in ts:
                    p0 = rows_0.clamp_min(1e-12)
                    pp = rows_p.clamp_min(1e-12)
                    h0 = float(-(p0 * p0.log()).sum())
                    hp = float(-(pp * pp.log()).sum())
                    out[step].append(hp > h0)
    return {t: np.asarray(v) for t, v in out.items()}


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["m"] < M_MAX]
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
        X = np.asarray(inst["X"], dtype=np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(7)
        err = ""
        try:
            Xt = torch.as_tensor(X, dtype=torch.float32, device=args.device)
            D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
            W = affinity_from_D(D, k=KNN)
            P = torch.as_tensor(
                W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30),
                dtype=torch.float32, device=args.device)
            if args.anchor_mode == "origin-local":
                # origin + its (n_local-1) nearest corpus points; all probe
                # edges pooled into ONE local fraction (finer resolution)
                order0 = np.argsort(D[0])
                anchors = [0] + [int(j) for j in order0[1:args.n_local]]
            else:
                r_c = np.linalg.norm(X - X.mean(0), axis=1)
                interior = np.where(r_c <= np.quantile(r_c, 0.5))[0]
                anchors = [0] + rng.choice(
                    interior, min(N_ANCHORS - 1, len(interior)),
                    replace=False).tolist()
            order = np.argsort(D[anchors], axis=1)
            edges = [(a, int(j)) for q, a in enumerate(anchors)
                     for j in order[q, 1:args.k_edge + 1]]
            fr = edge_fractions(P, edges, TS)
            scores = {}
            for t in TS:
                by_anchor = fr[t].reshape(len(anchors), args.k_edge).mean(axis=1)
                scores[f"frac_t{t}_origin"] = -float(by_anchor[0])
                if args.anchor_mode == "origin-local":
                    scores[f"frac_t{t}_mean"] = -float(fr[t].mean())
                else:
                    scores[f"frac_t{t}_mean"] = -float(by_anchor.mean())
        except Exception as e:
            scores = {f"frac_t{t}_{k}": float("nan")
                      for t in TS for k in ("origin", "mean")}  # noqa
            err = str(e)[:200]
            print(f"  [err] inst={i}: {err}", flush=True)

        pd.DataFrame([dict(instance=i, err=err, name=inst["name"],
                           dim=inst["dim"], noise=inst["noise"], m=inst["m"],
                           ks_true=inst["ks_true"], **scores)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} inst={i} d={inst['dim']} "
              f"({time.time()-t1:.0f}s) "
              + " ".join(f"{k}={v:+.2f}" for k, v in scores.items())
              + f" eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr, spearmanr
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("diffusing_fraction_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["instance"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    cols = [c for c in df.columns if c.startswith("frac_")]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        ok = df[df[col].notna()]
        rows = []
        for d, g in ok.groupby("dim"):
            if g[col].std() > 0:
                pos, neg = g[g.ks_true > 0], g[g.ks_true < 0]
                bal = (0.5 * ((pos[col] > 0.0 - 1e-9).mean()  # frac==0 -> "positive"
                              + (neg[col] < 0).mean())
                       if len(pos) and len(neg) else np.nan)
                rows.append(dict(dim=d,
                                 pearson=pearsonr(g[col], g.ks_true)[0],
                                 spearman=spearmanr(g[col], g.ks_true)[0],
                                 balsign=bal))
        s = pd.DataFrame(rows)
        print(f"=== {col} ===")
        print(s.round(2).to_string(index=False))
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--k-edge", type=int, default=K_EDGE)
    w.add_argument("--anchor-mode", default="interior",
                   choices=["interior", "origin-local"])
    w.add_argument("--n-local", type=int, default=5)
    w.add_argument("--out", default=None)
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        if args.out:
            global OUT_TPL
            OUT_TPL = args.out
        run_worker(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()
