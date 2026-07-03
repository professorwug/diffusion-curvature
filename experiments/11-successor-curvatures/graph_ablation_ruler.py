"""Graph-parameter ablation for the entropy ruler (colosseum battery).

The t*(d) map (composite v3-v5) showed d=2 wanting t>=128 on the k=10
adaptive kernel — i.e., per-step reach too small. Ablate the graph:

  adaptive_k{10,25,50} : adaptive gaussian kernel, k-th-NN bandwidths
  cak_{40,80,160}      : curvature-agnostic kernel (bandwidth bisected so the
                         one-step diffusion covers ~N points; fast
                         reimplementation of kernels.get_curvature_agnostic_graph
                         from the cached distance matrix, alpha=0)

Per (instance, graph): entropy of P^t rows at origin-local 5 anchors for
t in {1,2,4,8,16,32}, plus the one-step spread s1 (scale diagnostic).
Summarize: Pearson(z, ks_true) per (graph, t, dim); does the tuned kernel
unify t* across dims and lift d=2/d=3?

Usage:
  pixi run python graph_ablation_ruler.py run --worker-id K --num-workers W --device cuda:X
  pixi run python graph_ablation_ruler.py summarize
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

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
N_LOCAL = 5
TS = (1, 2, 4, 8, 16, 32)
KNNS = (10, 25, 50)
CAK_TARGETS = (40, 80, 160)

OUT_TPL = "processed_data/graph_ablation_w{wid}.csv"
OUT_MERGED = Path("processed_data/graph_ablation.csv")


def adaptive_W(D: np.ndarray, k: int, alpha: float = 1.0) -> np.ndarray:
    sig = np.partition(D, k, axis=1)[:, k]
    sig[sig <= 0] = sig[sig > 0].min() if (sig > 0).any() else 1.0
    W = 0.5 * (np.exp(-(D**2) / (2 * sig[None, :]**2)) / sig[None, :]
               + np.exp(-(D**2) / (2 * sig[:, None]**2)) / sig[:, None])
    if alpha:
        deg = 1.0 / (W.sum(axis=1) ** alpha)
        W = W * deg[:, None] * deg[None, :]
    W[W < 1e-8] = 0.0
    return W


def cak_W(D: np.ndarray, target: int, eps: float = 1e-10,
          iters: int = 25) -> tuple[np.ndarray, float]:
    """Curvature-agnostic kernel from a distance matrix: W = exp(-D^2/s^2),
    s = mean 1st-NN distance x neighbor_scale, bisected so the median
    one-step support size hits `target` (mirrors kernels.py, alpha=0)."""
    nn1 = np.partition(D, 1, axis=1)[:, 1]
    base = float(np.mean(nn1))

    def support(ns: float) -> float:
        W = np.exp(-(D**2) / (max(ns * base, 1e-12)**2))
        P = W / W.sum(axis=1, keepdims=True)
        return float(np.median((P > eps).sum(axis=1)))

    lo, hi = 0.05, 50.0
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if support(mid) > target:
            hi = mid
        else:
            lo = mid
    ns = float(np.sqrt(lo * hi))
    W = np.exp(-(D**2) / ((ns * base)**2))
    W[W < 1e-8] = 0.0
    return W, ns


def ruler_scores(W: np.ndarray, D: np.ndarray, anchors, device: str) -> dict:
    P = torch.as_tensor(W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30),
                        dtype=torch.float32, device=device)
    D_anch = torch.as_tensor(D[anchors], dtype=torch.float32, device=device)
    out = {}
    with torch.no_grad():
        rows = torch.zeros((len(anchors), W.shape[0]), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for step in range(1, max(TS) + 1):
            rows = rows @ P
            if step == 1:
                out["s1"] = float((rows * D_anch).sum(dim=1).median())
            if step in TS:
                pr = rows.clamp_min(1e-12)
                out[f"ent_t{step}"] = float((-(pr * pr.log()).sum(dim=1)).mean())
    return out


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["m"] < M_MAX]
    mine = [u for k, u in enumerate(cc) if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["instance", "graph"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} instances", flush=True)

    t0 = time.time()
    for prog, (i, inst) in enumerate(mine, 1):
        X = np.asarray(inst["X"], dtype=np.float64)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=args.device)
        D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
        rng = np.random.default_rng(7)
        order0 = np.argsort(D[0])
        anchors = [0] + [int(j) for j in order0[1:N_LOCAL]]

        graphs = {}
        for k in KNNS:
            graphs[f"adaptive_k{k}"] = (lambda kk=k: (adaptive_W(D, kk), np.nan))
        for tgt in CAK_TARGETS:
            graphs[f"cak_{tgt}"] = (lambda tt=tgt: cak_W(D, tt))

        for gname, builder in graphs.items():
            if (i, gname) in done:
                continue
            try:
                W, ns = builder()
                sc = ruler_scores(W, D, anchors, args.device)
                err = ""
            except Exception as e:
                sc, ns, err = {}, np.nan, str(e)[:200]
            pd.DataFrame([dict(
                instance=i, graph=gname, neighbor_scale=ns, err=err,
                dim=inst["dim"], noise=inst["noise"], ks_true=inst["ks_true"],
                **sc)]).to_csv(out, mode="a", index=False, header=not header)
            header = True
        if prog % 5 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("graph_ablation_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance", "graph"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)")
    print("\nmedian one-step spread s1 per (graph, dim):")
    print(df.pivot_table(index="graph", columns="dim", values="s1",
                         aggfunc="median").round(3).to_string())
    for gname, g in df.groupby("graph"):
        print(f"\n=== {gname}: Pearson(-ent_t, ks_true) per (t, dim) ===")
        rows = []
        for t in TS:
            col = f"ent_t{t}"
            row = {"t": t}
            for d, h in g.groupby("dim"):
                v = pd.to_numeric(h[col], errors="coerce")
                m = np.isfinite(v)
                row[f"d{d}"] = (round(pearsonr(-v[m], h.ks_true[m])[0], 2)
                                if m.sum() > 3 and v[m].std() > 0 else np.nan)
            rows.append(row)
        print(pd.DataFrame(rows).set_index("t").to_string())


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
