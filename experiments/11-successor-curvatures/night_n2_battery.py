"""N2 — battery test of the N1 survivors in the trajectory regime.

Channels per (colosseum instance m<15 d>=3, nt in {25, 100}), paired walks:
  resp_skew     : -skew of 32-edge response distribution at origin-local pool
                  (visited-cloud euclid graph)
  pair_contract : -mean log(d_k/d_0) over near cross-walk pairs (k=5),
                  pooled near the eval anchor (within r_loc) and globally
Flat packs: walk-subsampled planes at matched (d, nt), 12 reps.

Usage:
  pixi run python night_n2_battery.py run --worker-id K --num-workers W --device cuda:X
  pixi run python night_n2_battery.py flatpack --worker-id K --num-workers W --device cuda:X
  pixi run python night_n2_battery.py summarize
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
from scipy.stats import skew

from diffusion_curvature.trajectory_utils import subsample_trajectories
from benchmark_kmetric_colosseum import affinity_from_D
from ricci_flow_ladder import batched_edge_response
from diffusion_curvature.datasets import plane

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
NTS = (25, 100)
DIMS = (3, 4, 5, 6)
TRAJ_LEN = 50
KNN = 10
K_PAIR = 5
MIN_V = 60
N_REP_PACK = 12

OUT_TPL = "processed_data/night_n2_w{wid}.csv"
PACK_TPL = "processed_data/night_n2_flat_w{wid}.csv"


def n2_channels(X, traj_idx, device, rng):
    V = np.unique(traj_idx)
    nV = len(V)
    X_V = np.asarray(X, dtype=np.float64)[V]
    d0v = np.linalg.norm(X_V - X[0], axis=1)
    i0 = int(np.argmin(d0v))
    out = dict(coverage=nV / X.shape[0], eval_dist=float(d0v[i0]))
    if nV < MIN_V:
        return out

    Xt = torch.as_tensor(X_V, dtype=torch.float32, device=device)
    D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
    W10 = affinity_from_D(D, k=min(KNN, nV - 2))
    P10 = torch.as_tensor(
        W10 / np.maximum(W10.sum(axis=1, keepdims=True), 1e-30),
        dtype=torch.float32, device=device)
    order0 = np.argsort(D[i0])
    anchors = [i0] + [int(j) for j in order0[1:5]]
    sk = []
    for a in anchors:
        oa = np.argsort(D[a])
        edges = np.array([(a, int(j)) for j in oa[1:min(33, nV - 1)]])
        r = batched_edge_response(P10, edges, t=8)
        sk.append(skew(r))
    out["resp_skew"] = -float(np.nanmean(sk))

    # pair_contract near the eval anchor (r_loc = 30th pct of D[i0])
    coords = np.asarray(X, dtype=np.float64)[traj_idx]
    nt, Tp1 = traj_idx.shape
    flat = coords.reshape(-1, X.shape[1])
    walk_of = np.repeat(np.arange(nt), Tp1)
    t_of = np.tile(np.arange(Tp1), nt)
    valid = t_of < Tp1 - K_PAIR
    x0 = np.asarray(X, dtype=np.float64)[V[i0]]
    r_loc = np.quantile(d0v, 0.3)
    near_anchor = np.linalg.norm(flat - x0, axis=1) < r_loc
    cand = np.where(valid & near_anchor)[0]
    logs = []
    if len(cand) > 10:
        take = rng.choice(cand, min(1500, len(cand)), replace=False)
        at = torch.as_tensor(flat[take], dtype=torch.float32, device=device)
        ft = torch.as_tensor(flat, dtype=torch.float32, device=device)
        Dp = torch.cdist(at, ft).cpu().numpy()
        pos = Dp[Dp > 0]
        if pos.size:
            r_close = np.quantile(pos, 0.002)
            for qi, i in enumerate(take):
                js = np.where((Dp[qi] < r_close) & (Dp[qi] > 0)
                              & (walk_of != walk_of[i]) & valid)[0]
                for j in js[:2]:
                    d0 = Dp[qi, j]
                    dk = np.linalg.norm(
                        coords[walk_of[i], t_of[i] + K_PAIR]
                        - coords[walk_of[j], t_of[j] + K_PAIR])
                    if d0 > 0 and dk > 0:
                        logs.append(np.log(dk / d0))
    out["pair_contract"] = -float(np.mean(logs)) if len(logs) > 15 else np.nan
    out["pair_n"] = float(len(logs))
    return out


def _shard_loop(units, args, out_tpl, is_pack):
    out = Path(out_tpl.format(wid=args.worker_id))
    done = set()
    keycols = (["n_points", "dim", "n_traj", "rep"] if is_pack
               else ["instance", "n_traj"])
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=keycols)
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    mine = [u for k, u in enumerate(units) if k % args.num_workers == args.worker_id]
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
    for u in mine:
        if is_pack:
            (n, d, nt, rep) = u
            key = (n, d, nt, rep)
        else:
            (i, inst, nt) = u
            key = (i, nt)
        if key in done:
            continue
        err = ""
        ch = {}
        try:
            if is_pack:
                np.random.seed(30_000 + 1000 * d + 10 * nt + rep)
                X = np.hstack([plane(n, dim=d), np.zeros((n, 1))])
                meta = dict(n_points=n, dim=d, n_traj=nt, rep=rep)
                rng = np.random.default_rng(rep)
            else:
                X = np.asarray(inst["X"], dtype=np.float64)
                meta = dict(instance=i, n_traj=nt, dim=inst["dim"],
                            noise=inst["noise"], ks_true=inst["ks_true"])
                rng = np.random.default_rng(7)
            G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
            traj_idx = subsample_trajectories(
                G, n_trajectories=nt, length=TRAJ_LEN,
                rng=1000 + nt + (rep if is_pack else 0))
            ch = n2_channels(np.asarray(X, dtype=np.float64), traj_idx,
                             args.device, rng)
        except Exception as e:
            err = str(e)[:200]
        pd.DataFrame([dict(**meta, err=err, **ch)]).to_csv(
            out, mode="a", index=False, header=not header)
        header = True
    print(f"[w{args.worker_id}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "flatpack"):
        w = sub.add_parser(name)
        w.add_argument("--worker-id", type=int, default=0)
        w.add_argument("--num-workers", type=int, default=2)
        w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        instances = joblib.load(BATTERY_PATH)
        cc = [(i, inst) for i, inst in enumerate(instances)
              if inst["dataset"] == "colosseum" and inst["m"] < M_MAX
              and inst["dim"] in DIMS]
        units = [(i, inst, nt) for (i, inst), nt in itertools.product(cc, NTS)]
        _shard_loop(units, args, OUT_TPL, is_pack=False)
    elif args.cmd == "flatpack":
        units = list(itertools.product((3000,), DIMS, NTS, range(N_REP_PACK)))
        _shard_loop(units, args, PACK_TPL, is_pack=True)
    else:
        from scipy.stats import pearsonr
        df = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob("night_n2_w*.csv"))],
                       ignore_index=True).drop_duplicates(
            ["instance", "n_traj"], keep="last")
        flat = pd.concat([pd.read_csv(p) for p in
                          sorted(Path("processed_data").glob("night_n2_flat_w*.csv"))],
                         ignore_index=True)
        df.to_csv("processed_data/night_n2.csv", index=False)
        for ch in ("resp_skew", "pair_contract"):
            ref = flat.groupby(["dim", "n_traj"])[ch].agg(["mean", "std"])
            print(f"\n=== {ch}: Pearson | balanced-sign-at-flat-zero per (nt, dim) ===")
            rows = []
            for nt in NTS:
                row = {"n_traj": nt}
                for d in DIMS:
                    g = df[(df.n_traj == nt) & (df.dim == d)]
                    v = pd.to_numeric(g[ch], errors="coerce")
                    m = np.isfinite(v)
                    if m.sum() > 5 and v[m].std() > 0:
                        r = pearsonr(v[m], g.ks_true[m])[0]
                        mu = ref.loc[(d, nt), "mean"]
                        pos = m & (g.ks_true > 0)
                        neg = m & (g.ks_true < 0)
                        bal = (0.5 * ((v[pos] > mu).mean() + (v[neg] < mu).mean())
                               if pos.any() and neg.any() else np.nan)
                        row[f"d{d}"] = f"{r:+.2f}|{bal:.2f}"
                    else:
                        row[f"d{d}"] = "  -  "
                rows.append(row)
            print(pd.DataFrame(rows).set_index("n_traj").to_string())


if __name__ == "__main__":
    main()
