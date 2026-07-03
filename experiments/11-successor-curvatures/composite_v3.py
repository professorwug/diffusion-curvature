"""Composite v3: ruler-channel study — {entropy, Wasserstein spread} x t grid.

Hypothesis (user): the low-dim inversion of the entropy ruler is a boundary
effect — auto-t overspreads into the boundary at d=2-3 where sampling is
fine. Test: compute both ruler functionals at t in {2,4,8,16,32} AND auto-t;
if small-t repairs d<=3 ordering, boundary confirmed.

Channels per instance: kappa_plus + frac (as v2) + ruler grid
  ent_t{T}     : Shannon entropy of P^t rows, origin-local 5 anchors
  wspread_t{T} : sum_a mu(a) D_euc(anchor, a)  (Wasserstein-1 spread)
(auto rows recorded as ent_auto / wspread_auto with t_auto logged.)

Summarize: per ruler variant, ordering quality (Pearson per dim) and the
K_cal / K_gated fusions against S_diff.

Usage: same flatpack/run/summarize + army pattern as composite_signed_colosseum.
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
from successor_ricci_ladder import _auto_t_dense

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
KNN = 10
T_ORC = 8
T_FRAC = 16
K_EDGE = 16
N_LOCAL = 5
N_REP = 30
TS_RULER = (2, 4, 8, 16, 32)
FLAT_DIMS = (2, 3, 4, 5, 6)
FLAT_NS = (2000, 3000)

OUT_RUN_TPL = "processed_data/composite3_run_w{wid}.csv"
OUT_FLAT_TPL = "processed_data/composite3_flat_w{wid}.csv"
OUT_MERGED = Path("processed_data/composite_signed_v3.csv")


def channels(X: np.ndarray, device: str) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    est = WassersteinSignedCurvature(t=T_ORC, knn=KNN, n_pairs=8, seed=0,
                                     compute_midpoint=False)
    est.fit(X=X, idx=[0])
    out = dict(kappa_plus=float(est.orc_[0]))

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
    out["frac"] = float(fr[T_FRAC].mean())

    t_auto = _auto_t_dense(W, D, anchors)
    out["t_auto"] = t_auto
    D_anch = torch.as_tensor(D[anchors], dtype=torch.float32, device=device)
    checkpoints = sorted(set(TS_RULER) | {t_auto})
    with torch.no_grad():
        rows = torch.zeros((len(anchors), X.shape[0]), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for step in range(1, max(checkpoints) + 1):
            rows = rows @ P
            if step in checkpoints:
                pr = rows.clamp_min(1e-12)
                H = float((-(pr * pr.log()).sum(dim=1)).mean())
                Wsp = float((rows * D_anch).sum(dim=1).mean())
                if step in TS_RULER:
                    out[f"ent_t{step}"] = H
                    out[f"wspread_t{step}"] = Wsp
                if step == t_auto:
                    out["ent_auto"] = H
                    out["wspread_auto"] = Wsp
    return out


RULERS = ([f"ent_t{t}" for t in TS_RULER] + ["ent_auto"]
          + [f"wspread_t{t}" for t in TS_RULER] + ["wspread_auto"])
NAN_CH = {**{k: np.nan for k in
             ("kappa_plus", "frac", *RULERS)}, "t_auto": -1}


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
            ch, err = channels(X, args.device), ""
        except Exception as e:
            ch, err = dict(NAN_CH), str(e)[:200]
        pd.DataFrame([dict(n_points=n, dim=d, rep=rep, err=err, **ch)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
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
            ch, err = channels(np.asarray(inst["X"], dtype=np.float64),
                               args.device), ""
        except Exception as e:
            ch, err = dict(NAN_CH), str(e)[:200]
        pd.DataFrame([dict(
            instance=i, dataset=inst["dataset"], err=err, name=inst["name"],
            dim=inst["dim"], noise=inst["noise"], m=inst["m"],
            shape=inst.get("shape", ""), ks_true=inst["ks_true"],
            n_points=np.asarray(inst["X"]).shape[0], **ch)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr
    flat = pd.concat([pd.read_csv(p) for p in
                      sorted(Path("processed_data").glob("composite3_flat_w*.csv"))],
                     ignore_index=True).drop_duplicates(
        ["n_points", "dim", "rep"], keep="last")
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("composite3_run_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance"], keep="last").reset_index(drop=True)

    # base channels
    agg = {f"mu_{c}": (c, "mean") for c in ("kappa_plus", "frac", *RULERS)}
    agg.update({f"sd_{c}": (c, "std") for c in ("kappa_plus", "frac", *RULERS)})
    ref = flat.groupby(["n_points", "dim"]).agg(**agg).reset_index()
    df = df.merge(ref, on=["n_points", "dim"], how="left")
    df["z_plus"] = (df.kappa_plus - df.mu_kappa_plus) / df.sd_kappa_plus.clip(lower=1e-4)
    df["z_frac"] = (df.frac - df.mu_frac) / df.sd_frac.clip(lower=1e-3)
    df["S_diff"] = df.z_plus - df.z_frac
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)")
    print(f"t_auto by dim (colosseum): "
          f"{df[df.dataset=='colosseum'].groupby('dim').t_auto.median().to_dict()}\n")

    cc = df[df.dataset == "colosseum"]
    print("=== RULER ORDERING: Pearson(z_ruler, ks_true) per dim (colosseum) ===")
    for r in RULERS:
        z = (df[f"mu_{r}"] - df[r]) / df[f"sd_{r}"].clip(lower=1e-6)
        df[f"z_{r}"] = z
        row = []
        for d, g in cc.groupby("dim"):
            zz = z.loc[g.index]
            m = np.isfinite(zz)
            v = pearsonr(zz[m], g.ks_true[m])[0] if zz[m].std() > 0 else np.nan
            row.append(f"d{d}: {v:+.2f}")
        print(f"{r:<14} " + "  ".join(row))

    # best ruler per the low-dim cells -> fusion report for top-3 rulers
    print("\n=== K_gated fusion (ruler ordering + S_diff zero/sign) — top rulers ===")
    def score_ruler(r):
        vals = []
        for d, g in cc.groupby("dim"):
            zz = df.loc[g.index, f"z_{r}"]
            m = np.isfinite(zz)
            if zz[m].std() > 0:
                vals.append(pearsonr(zz[m], g.ks_true[m])[0])
        return np.nanmean(vals)
    top = sorted(RULERS, key=score_ruler, reverse=True)[:3]
    for r in top:
        zr = df[f"z_{r}"]
        def _offset(g):
            z, s_ = g[f"z_{r}"].to_numpy(), g.S_diff.to_numpy()
            w = np.abs(s_)
            cands = np.unique(z)
            cands = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
            return max(((float((w * (np.sign(z - c) == np.sign(s_))).sum()), float(c))
                        for c in cands), default=(0, 0.0))[1]
        offs = {k: _offset(g) for k, g in df.groupby(["n_points", "dim"])}
        c_off = np.array([offs[(n, d)] for n, d in zip(df.n_points, df.dim)])
        K = zr - c_off
        disagree = (np.sign(K) != np.sign(df.S_diff)) & (df.S_diff.abs() > 1)
        Kg = np.where(disagree, np.abs(K) * np.sign(df.S_diff), K)
        row = []
        for d, g in cc.groupby("dim"):
            idx = g.index.to_numpy()
            kg = Kg[idx]
            kt = g.ks_true.to_numpy()
            m = np.isfinite(kg)
            r_p = (pearsonr(kg[m], kt[m])[0]
                   if m.sum() > 3 and np.std(kg[m]) > 0 else np.nan)
            pos, neg = m & (kt > 0), m & (kt < 0)
            bal = (0.5 * ((kg[pos] > 0).mean() + (kg[neg] < 0).mean())
                   if pos.any() and neg.any() else np.nan)
            row.append(f"d{d}: {r_p:+.2f}|{bal:.2f}")
        print(f"K_gated[{r}] " + "  ".join(row))


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
