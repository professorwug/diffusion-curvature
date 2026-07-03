"""Composite v6 — the assembly: cak_640 entropy ruler + S_diff zero/sign.

Channels per instance (origin-local):
  kappa_plus  : Diffusion ORC t=8, adaptive k=10 (positive-sign channel)
  frac        : diffusing-edge fraction t=16, adaptive k=10 (negative-sign)
  ent_cak_t{2,4,8} : diffusion entropy on the curvature-agnostic kernel
                     (one-step support ~640) — the ruler (graph ablation
                     winner: Pearson 0.56-0.79 all dims at t=4)
  ent_k10_t8  : fine-kernel ruler (d=6 specialist, 0.72)

Fusion: S_diff = z+ - z_f (sign/zero); K_gated[ruler] = flat-pack z-scored
ruler with per-(n,d) offset aligned to S_diff and confident-sign override.

Usage: flatpack/run/summarize; army as usual.
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
from graph_ablation_ruler import cak_W

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
KNN = 10
T_ORC = 8
T_FRAC = 16
K_EDGE = 16
N_LOCAL = 5
N_REP = 30
CAK_N = 640
TS_CAK = (2, 4, 8)
FLAT_DIMS = (2, 3, 4, 5, 6)
FLAT_NS = (2000, 3000)

OUT_RUN_TPL = "processed_data/composite6_run_w{wid}.csv"
OUT_FLAT_TPL = "processed_data/composite6_flat_w{wid}.csv"
OUT_MERGED = Path("processed_data/composite_signed_v6.csv")

RULERS = [f"ent_cak_t{t}" for t in TS_CAK] + ["ent_k10_t8"]
NAN_CH = {k: np.nan for k in ("kappa_plus", "frac", *RULERS)}


def channels(X: np.ndarray, device: str) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    est = WassersteinSignedCurvature(t=T_ORC, knn=KNN, n_pairs=8, seed=0,
                                     compute_midpoint=False)
    est.fit(X=X, idx=[0])
    out = dict(kappa_plus=float(est.orc_[0]))

    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
    order0 = np.argsort(D[0])
    anchors = [0] + [int(j) for j in order0[1:N_LOCAL]]

    def ent_grid(W, ts):
        P = torch.as_tensor(
            W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30),
            dtype=torch.float32, device=device)
        res = {}
        with torch.no_grad():
            rows = torch.zeros((len(anchors), X.shape[0]), device=device)
            for q, a in enumerate(anchors):
                rows[q, a] = 1.0
            for step in range(1, max(ts) + 1):
                rows = rows @ P
                if step in ts:
                    pr = rows.clamp_min(1e-12)
                    res[step] = float((-(pr * pr.log()).sum(dim=1)).mean())
        return res, P

    W10 = affinity_from_D(D, k=KNN)
    ent10, P10 = ent_grid(W10, (T_ORC,))
    out["ent_k10_t8"] = ent10[T_ORC]
    order = np.argsort(D[anchors], axis=1)
    edges = [(a, int(j)) for q, a in enumerate(anchors)
             for j in order[q, 1:K_EDGE + 1]]
    fr = edge_fractions(P10, edges, (T_FRAC,))
    out["frac"] = float(fr[T_FRAC].mean())

    W_cak, _ = cak_W(D, CAK_N)
    ent_cak, _ = ent_grid(W_cak, TS_CAK)
    for t in TS_CAK:
        out[f"ent_cak_t{t}"] = ent_cak[t]
    return out


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
    for (n, d, rep) in mine:
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
    print(f"[w{args.worker_id}] flatpack done", flush=True)


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
        if prog % 10 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] battery done", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    flat = pd.concat([pd.read_csv(p) for p in
                      sorted(Path("processed_data").glob("composite6_flat_w*.csv"))],
                     ignore_index=True).drop_duplicates(
        ["n_points", "dim", "rep"], keep="last")
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("composite6_run_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance"], keep="last").reset_index(drop=True)

    chans = ("kappa_plus", "frac", *RULERS)
    agg = {f"mu_{c}": (c, "mean") for c in chans}
    agg.update({f"sd_{c}": (c, "std") for c in chans})
    ref = flat.groupby(["n_points", "dim"]).agg(**agg).reset_index()
    df = df.merge(ref, on=["n_points", "dim"], how="left")
    df["z_plus"] = (df.kappa_plus - df.mu_kappa_plus) / df.sd_kappa_plus.clip(lower=1e-4)
    df["z_frac"] = (df.frac - df.mu_frac) / df.sd_frac.clip(lower=1e-3)
    df["S_diff"] = df.z_plus - df.z_frac
    for r in RULERS:
        df[f"z_{r}"] = (df[f"mu_{r}"] - df[r]) / df[f"sd_{r}"].clip(lower=1e-6)
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")

    def gated(r):
        def _offset(g):
            z, s_ = g[f"z_{r}"].to_numpy(), g.S_diff.to_numpy()
            w = np.abs(s_)
            cands = np.unique(z[np.isfinite(z)])
            cands = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
            return max(((float((w * (np.sign(z - c) == np.sign(s_))).sum()),
                         float(c)) for c in cands), default=(0, 0.0))[1]
        offs = {k: _offset(g) for k, g in df.groupby(["n_points", "dim"])}
        c_off = np.array([offs[(n, d)] for n, d in zip(df.n_points, df.dim)])
        K = df[f"z_{r}"].to_numpy() - c_off
        dis = (np.sign(K) != np.sign(df.S_diff)) & (df.S_diff.abs() > 1)
        return np.where(dis, np.abs(K) * np.sign(df.S_diff), K)

    estimators = {"S_diff": df.S_diff.to_numpy()}
    for r in RULERS:
        estimators[f"K_gated[{r}]"] = gated(r)

    cc_mask = (df.dataset == "colosseum").to_numpy()
    print("=== COLOSSEUM per dim: pearson | spearman | balanced sign AT ZERO ===")
    for name, K in estimators.items():
        row = []
        for d in FLAT_DIMS:
            m = cc_mask & (df.dim == d).to_numpy() & np.isfinite(K)
            kt = df.ks_true.to_numpy()
            r_p = pearsonr(K[m], kt[m])[0] if m.sum() > 3 else np.nan
            s_p = spearmanr(K[m], kt[m])[0] if m.sum() > 3 else np.nan
            pos, neg = m & (kt > 0), m & (kt < 0)
            bal = (0.5 * ((K[pos] > 0).mean() + (K[neg] < 0).mean())
                   if pos.any() and neg.any() else np.nan)
            row.append(f"d{d}: {r_p:+.2f}|{s_p:+.2f}|{bal:.2f}")
        print(f"{name:<22} " + "  ".join(row))

    ss_mask = ((df.dataset == "sadspheres") & (df.ks_true != 0)).to_numpy()
    print("\n=== SADSPHERES per dim: AUC | sign-at-zero ===")
    for name, K in estimators.items():
        row = []
        for d in FLAT_DIMS:
            m = ss_mask & (df.dim == d).to_numpy() & np.isfinite(K)
            kt = df.ks_true.to_numpy()
            auc = (roc_auc_score((kt[m] > 0).astype(int), K[m])
                   if m.sum() > 3 and len(set(kt[m] > 0)) > 1 else np.nan)
            sign = float(np.mean(np.sign(K[m]) == np.sign(kt[m]))) if m.any() else np.nan
            row.append(f"d{d}: {auc:.2f}|{sign:.2f}")
        print(f"{name:<22} " + "  ".join(row))


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
