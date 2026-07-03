"""Composite signed estimator: positive channel (Diffusion ORC) + negative
channel (diffusing-edge fraction), fused against a flat pack.

Channels per instance (full pointcloud, origin-anchored):
  kappa_plus : W1-contraction Diffusion ORC at t=8 (validated K>0 instrument)
  frac       : diffusing-edge fraction, origin-local 5 anchors x 16 edges,
               t=16 (validated K<0 instrument; hard floor at 0 on K>0)

Flat pack: N_REP planes per (n_points, dim) give (mu, sigma) per channel —
two scalars of "how noisy is quiet"; NOT a per-manifold comparison space.

Combiners (in summarize):
  S_diff = z_plus - z_frac
  S_sel  = +z_plus if z_plus > max(z_frac, tau) else -z_frac if z_frac >
           max(z_plus, tau) else 0   (tau = 1; abstention semantics)

Usage:
  pixi run python composite_signed_colosseum.py flatpack --worker-id K --num-workers W --device cuda:X
  pixi run python composite_signed_colosseum.py run --worker-id K --num-workers W --device cuda:X
  pixi run python composite_signed_colosseum.py summarize
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
import torch

from diffusion_curvature.datasets import plane
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from benchmark_kmetric_colosseum import affinity_from_D
from diffusing_fraction_colosseum import edge_fractions

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
KNN = 10
T_ORC = 8
T_FRAC = 16
K_EDGE = 16
N_LOCAL = 5
N_REP = 30
FLAT_DIMS = (2, 3, 4, 5, 6)
FLAT_NS = (2000, 3000)   # sadspheres / colosseum sizes
TAU_SEL = 1.0

OUT_RUN_TPL = "processed_data/composite_run_w{wid}.csv"
OUT_FLAT_TPL = "processed_data/composite_flat_w{wid}.csv"
OUT_MERGED = Path("processed_data/composite_signed.csv")


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


def channels(X: np.ndarray, device: str) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    est = WassersteinSignedCurvature(t=T_ORC, knn=KNN, n_pairs=8, seed=0,
                                     compute_midpoint=False)
    est.fit(X=X, idx=[0])
    kappa_plus = float(est.orc_[0])

    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
    W = affinity_from_D(D, k=KNN)
    P = torch.as_tensor(W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30),
                        dtype=torch.float32, device=device)
    order0 = np.argsort(D[0])
    anchors = [0] + [int(j) for j in order0[1:N_LOCAL]]
    order = np.argsort(D[anchors], axis=1)
    edges = [(a, int(j)) for q, a in enumerate(anchors)
             for j in order[q, 1:K_EDGE + 1]]
    fr = edge_fractions(P, edges, (T_FRAC,))
    return dict(kappa_plus=kappa_plus, frac=float(fr[T_FRAC].mean()))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def run_flatpack(args) -> None:
    units = list(itertools.product(FLAT_NS, FLAT_DIMS, range(N_REP)))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(OUT_FLAT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["n_points", "dim", "rep"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] flatpack: {len(mine)} units", flush=True)
    t0 = time.time()
    for prog, (n, d, rep) in enumerate(mine, 1):
        if (n, d, rep) in done:
            continue
        np.random.seed(10_000 + 100 * d + rep)
        X = np.hstack([plane(n, dim=d), np.zeros((n, 1))])
        try:
            ch = channels(X, args.device)
            err = ""
        except Exception as e:
            ch, err = dict(kappa_plus=np.nan, frac=np.nan), str(e)[:200]
        pd.DataFrame([dict(n_points=n, dim=d, rep=rep, err=err, **ch)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
        if prog % 10 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_battery(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    todo = [(i, inst) for i, inst in enumerate(instances)
            if inst["dataset"] == "sadspheres"
            or (inst["dataset"] == "colosseum" and inst["m"] < M_MAX)]
    mine = [u for k, u in enumerate(todo) if k % args.num_workers == args.worker_id]
    out = Path(OUT_RUN_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        done = set(pd.read_csv(out, usecols=["instance"]).instance)
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] battery: {len(mine)} instances", flush=True)
    t0 = time.time()
    for prog, (i, inst) in enumerate(mine, 1):
        if i in done:
            continue
        try:
            ch = channels(np.asarray(inst["X"], dtype=np.float64), args.device)
            err = ""
        except Exception as e:
            ch, err = dict(kappa_plus=np.nan, frac=np.nan), str(e)[:200]
        pd.DataFrame([dict(
            instance=i, dataset=inst["dataset"], err=err, name=inst["name"],
            dim=inst["dim"], noise=inst["noise"], m=inst["m"],
            shape=inst.get("shape", ""), ks_true=inst["ks_true"],
            n_points=np.asarray(inst["X"]).shape[0], **ch)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
        if prog % 10 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    flat = pd.concat([pd.read_csv(p) for p in
                      sorted(Path("processed_data").glob("composite_flat_w*.csv"))],
                     ignore_index=True).drop_duplicates(
        ["n_points", "dim", "rep"], keep="last")
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("composite_run_w*.csv"))],
                   ignore_index=True).drop_duplicates(["instance"], keep="last")

    ref = flat.groupby(["n_points", "dim"]).agg(
        mu_k=("kappa_plus", "mean"), sd_k=("kappa_plus", "std"),
        mu_f=("frac", "mean"), sd_f=("frac", "std")).reset_index()
    ref["sd_k"] = ref.sd_k.clip(lower=1e-4)
    ref["sd_f"] = ref.sd_f.clip(lower=1e-3)
    df = df.merge(ref, on=["n_points", "dim"], how="left")
    df["z_plus"] = (df.kappa_plus - df.mu_k) / df.sd_k
    df["z_frac"] = (df.frac - df.mu_f) / df.sd_f
    df["S_diff"] = df.z_plus - df.z_frac
    sel_pos = (df.z_plus > df.z_frac) & (df.z_plus > TAU_SEL)
    sel_neg = (df.z_frac >= df.z_plus) & (df.z_frac > TAU_SEL)
    df["S_sel"] = np.where(sel_pos, df.z_plus, np.where(sel_neg, -df.z_frac, 0.0))
    df["kappa_only"] = df.z_plus
    df["frac_only"] = -df.z_frac
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")

    scores = ["kappa_only", "frac_only", "S_diff", "S_sel"]
    cc = df[(df.dataset == "colosseum") & np.isfinite(df.S_diff)]
    print("=== COLOSSEUM per dim: pearson | spearman | balanced sign AT ZERO ===")
    for sc in scores:
        row = []
        for d, g in cc.groupby("dim"):
            pos, neg = g[g.ks_true > 0], g[g.ks_true < 0]
            bal = (0.5 * ((pos[sc] > 0).mean() + (neg[sc] < 0).mean())
                   if len(pos) and len(neg) else np.nan)
            r = pearsonr(g[sc], g.ks_true)[0] if g[sc].std() > 0 else np.nan
            s = spearmanr(g[sc], g.ks_true)[0] if g[sc].std() > 0 else np.nan
            row.append(f"d{d}: {r:+.2f}|{s:+.2f}|{bal:.2f}")
        print(f"{sc:<11} " + "  ".join(row))
    ss = df[(df.dataset == "sadspheres") & np.isfinite(df.S_diff)
            & (df.ks_true != 0)]
    print("\n=== SADSPHERES per dim: AUC | sign-at-zero ===")
    for sc in scores:
        row = []
        for d, g in ss.groupby("dim"):
            auc = (roc_auc_score((g.ks_true > 0).astype(int), g[sc])
                   if g.ks_true.nunique() > 1 else np.nan)
            sign = float(np.mean(np.sign(g[sc]) == np.sign(g.ks_true)))
            row.append(f"d{d}: {auc:.2f}|{sign:.2f}")
        print(f"{sc:<11} " + "  ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("flatpack", "run"):
        w = sub.add_parser(name)
        w.add_argument("--worker-id", type=int, default=0)
        w.add_argument("--num-workers", type=int, default=2)
        w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "flatpack":
        run_flatpack(args)
    elif args.cmd == "run":
        run_battery(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()
