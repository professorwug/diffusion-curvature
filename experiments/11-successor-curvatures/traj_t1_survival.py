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
    for name in ("run", "flatpack"):
        w = sub.add_parser(name)
        w.add_argument("--worker-id", type=int, default=0)
        w.add_argument("--num-workers", type=int, default=2)
        w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    sub.add_parser("summarize-t2")
    args = ap.parse_args()
    if args.cmd == "run":
        run_worker(args)
    elif args.cmd == "flatpack":
        run_flatpack(args)
    elif args.cmd == "summarize-t2":
        run_summarize_t2()
    else:
        run_summarize()




# ---------------------------------------------------------------------------
# T2: coverage-matched flat packs (planes -> walks -> visited channels) and
# the trajectory-regime composite fusion over the T1 battery channels.
# ---------------------------------------------------------------------------

N_REP_PACK = 15
PACK_TPL = "processed_data/traj_t2_flat_w{wid}.csv"


def run_flatpack(args) -> None:
    from diffusion_curvature.datasets import plane
    units = list(itertools.product((3000,), (2, 3, 4, 5, 6), NTS,
                                   range(N_REP_PACK)))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(PACK_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["n_points", "dim", "n_traj", "rep"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] T2 flatpack: {len(mine)} units", flush=True)
    for (n, d, nt, rep) in mine:
        if (n, d, nt, rep) in done:
            continue
        np.random.seed(20_000 + 1000 * d + 10 * nt + rep)
        X = np.hstack([plane(n, dim=d), np.zeros((n, 1))])
        err = ""
        ch: dict[str, float] = {k: np.nan for k in CH_KEYS}
        coverage = np.nan
        try:
            G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
            traj_idx = subsample_trajectories(
                G, n_trajectories=nt, length=TRAJ_LEN, rng=1000 + nt + rep)
            V = np.unique(traj_idx)
            coverage = len(V) / n
            X_V = np.asarray(X, dtype=np.float64)[V]
            i0 = int(np.argmin(np.linalg.norm(X_V - X[0], axis=1)))
            if len(V) >= MIN_V:
                ch = traj_channels(X_V, i0, args.device)
        except Exception as e:
            err = str(e)[:200]
        pd.DataFrame([dict(n_points=n, dim=d, n_traj=nt, rep=rep, err=err,
                           coverage=coverage, **ch)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
    print(f"[w{args.worker_id}] flatpack done", flush=True)


def run_summarize_t2() -> None:
    from scipy.stats import pearsonr, spearmanr
    flat = pd.concat([pd.read_csv(p) for p in
                      sorted(Path("processed_data").glob("traj_t2_flat_w*.csv"))],
                     ignore_index=True).drop_duplicates(
        ["n_points", "dim", "n_traj", "rep"], keep="last")
    df = pd.read_csv(OUT_MERGED).reset_index(drop=True)

    agg = {f"mu_{c}": (c, "mean") for c in CH_KEYS}
    agg.update({f"sd_{c}": (c, "std") for c in CH_KEYS})
    ref = flat.groupby(["dim", "n_traj"]).agg(**agg).reset_index()
    df = df.merge(ref, on=["dim", "n_traj"], how="left")
    df["z_plus"] = (df.kappa_plus - df.mu_kappa_plus) / df.sd_kappa_plus.clip(lower=1e-4)
    df["z_frac"] = (df.frac - df.mu_frac) / df.sd_frac.clip(lower=1e-3)
    df["S_diff"] = df.z_plus - df.z_frac
    for r in [f"ent_cak_t{t}" for t in TS_CAK]:
        df[f"z_{r}"] = (df[f"mu_{r}"] - df[r]) / df[f"sd_{r}"].clip(lower=1e-6)

    def gated(r):
        def _offset(g):
            z, s_ = g[f"z_{r}"].to_numpy(), g.S_diff.to_numpy()
            m = np.isfinite(z) & np.isfinite(s_)
            z, s_ = z[m], s_[m]
            if not len(z):
                return 0.0
            w = np.abs(s_)
            cands = np.unique(z)
            cands = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
            return max(((float((w * (np.sign(z - c) == np.sign(s_))).sum()),
                         float(c)) for c in cands), default=(0, 0.0))[1]
        offs = {k: _offset(g) for k, g in df.groupby(["dim", "n_traj"])}
        c_off = np.array([offs[(d, nt)] for d, nt in zip(df.dim, df.n_traj)])
        K = df[f"z_{r}"].to_numpy() - c_off
        dis = (np.sign(K) != np.sign(df.S_diff)) & (df.S_diff.abs() > 1)
        return np.where(dis, np.abs(K) * np.sign(df.S_diff), K)

    ests = {"S_diff": df.S_diff.to_numpy(),
            "K_gated[cak_t4]": gated("ent_cak_t4"),
            "K_gated[cak_t8]": gated("ent_cak_t8")}
    for name, K in ests.items():
        print(f"\n=== T2 {name}: pearson | balanced sign AT ZERO per (n_traj, dim) ===")
        rows = []
        for nt in NTS:
            row = {"n_traj": nt}
            for d in (2, 3, 4, 5, 6):
                m = ((df.n_traj == nt) & (df.dim == d)).to_numpy() & np.isfinite(K)
                kt = df.ks_true.to_numpy()
                if m.sum() > 5 and np.std(K[m]) > 0:
                    r_p = pearsonr(K[m], kt[m])[0]
                    pos, neg = m & (kt > 0), m & (kt < 0)
                    bal = (0.5 * ((K[pos] > 0).mean() + (K[neg] < 0).mean())
                           if pos.any() and neg.any() else np.nan)
                    row[f"d{d}"] = f"{r_p:+.2f}|{bal:.2f}"
                else:
                    row[f"d{d}"] = "  -  "
            rows.append(row)
        print(pd.DataFrame(rows).set_index("n_traj").to_string())
if __name__ == "__main__":
    main()
