"""N5 — Toponogov channel on the trajectory battery (candidate kappa+ upgrade).

Per (colosseum instance m<15 d>=3, nt in {25, 100}), paired walks with T1:
topo score at the eval anchor on the visited cloud. Flat packs from
walk-subsampled planes (12 reps). Summarize compares to kappa_plus (T1).

Usage: run / flatpack / summarize (army pattern).
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.datasets import plane
from diffusion_curvature.trajectory_utils import subsample_trajectories
from night_n4_toponogov import toponogov_score

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
NTS = (25, 100)
DIMS = (3, 4, 5, 6)
TRAJ_LEN = 50
KNN = 10
MIN_V = 100
N_REP_PACK = 12

OUT_TPL = "processed_data/night_n5_w{wid}.csv"
PACK_TPL = "processed_data/night_n5_flat_w{wid}.csv"


def unit_score(X, nt, rng_walk, rng):
    G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
    ti = subsample_trajectories(G, n_trajectories=nt, length=TRAJ_LEN,
                                rng=rng_walk)
    V = np.unique(ti)
    if len(V) < MIN_V:
        return dict(coverage=len(V) / X.shape[0], topo=np.nan)
    X_V = np.asarray(X, dtype=np.float64)[V]
    i0 = int(np.argmin(np.linalg.norm(X_V - np.asarray(X)[0], axis=1)))
    return dict(coverage=len(V) / X.shape[0],
                topo=toponogov_score(X_V, i0, rng))


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

    if args.cmd in ("run", "flatpack"):
        is_pack = args.cmd == "flatpack"
        tpl = PACK_TPL if is_pack else OUT_TPL
        if is_pack:
            units = list(itertools.product(DIMS, NTS, range(N_REP_PACK)))
        else:
            instances = joblib.load(BATTERY_PATH)
            cc = [(i, inst) for i, inst in enumerate(instances)
                  if inst["dataset"] == "colosseum" and inst["m"] < M_MAX
                  and inst["dim"] in DIMS]
            units = [(i, inst, nt) for (i, inst), nt in
                     itertools.product(cc, NTS)]
        out = Path(tpl.format(wid=args.worker_id))
        done = set()
        keyc = ["dim", "n_traj", "rep"] if is_pack else ["instance", "n_traj"]
        if out.exists() and out.stat().st_size > 0:
            done = set(map(tuple, pd.read_csv(out, usecols=keyc).values))
        header = out.exists() and out.stat().st_size > 0
        mine = [u for k, u in enumerate(units)
                if k % args.num_workers == args.worker_id]
        print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
        for u in mine:
            if is_pack:
                d, nt, rep = u
                key = (d, nt, rep)
                np.random.seed(40_000 + 1000 * d + 10 * nt + rep)
                X = np.hstack([plane(3000, dim=d), np.zeros((3000, 1))])
                meta = dict(dim=d, n_traj=nt, rep=rep)
                rw, rng = 1000 + nt + rep, np.random.default_rng(rep)
            else:
                i, inst, nt = u
                key = (i, nt)
                X = np.asarray(inst["X"], dtype=np.float64)
                meta = dict(instance=i, n_traj=nt, dim=inst["dim"],
                            noise=inst["noise"], ks_true=inst["ks_true"])
                rw, rng = 1000 + nt, np.random.default_rng(7)
            if key in done:
                continue
            err, sc = "", {}
            try:
                sc = unit_score(X, nt, rw, rng)
            except Exception as e:
                err = str(e)[:200]
            pd.DataFrame([dict(**meta, err=err, **sc)]).to_csv(
                out, mode="a", index=False, header=not header)
            header = True
        print(f"[w{args.worker_id}] done", flush=True)
    else:
        from scipy.stats import pearsonr
        df = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob("night_n5_w*.csv"))],
                       ignore_index=True).drop_duplicates(
            ["instance", "n_traj"], keep="last")
        flat = pd.concat([pd.read_csv(p) for p in
                          sorted(Path("processed_data").glob(
                              "night_n5_flat_w*.csv"))], ignore_index=True)
        df.to_csv("processed_data/night_n5.csv", index=False)
        t1 = pd.read_csv("processed_data/traj_t1.csv")
        df = df.merge(t1[["instance", "n_traj", "kappa_plus"]],
                      on=["instance", "n_traj"], how="left")
        ref = flat.groupby(["dim", "n_traj"]).topo.agg(["mean", "std"])
        print("=== topo vs kappa_plus: Pearson (topo|kappa) + topo balanced-sign@flat-zero ===")
        rows = []
        for nt in NTS:
            row = {"n_traj": nt}
            for d in DIMS:
                g = df[(df.n_traj == nt) & (df.dim == d)]
                v = pd.to_numeric(g.topo, errors="coerce")
                k = pd.to_numeric(g.kappa_plus, errors="coerce")
                m, mk = np.isfinite(v), np.isfinite(k)
                r = (pearsonr(v[m], g.ks_true[m])[0]
                     if m.sum() > 5 and v[m].std() > 0 else np.nan)
                rk = (pearsonr(k[mk], g.ks_true[mk])[0]
                      if mk.sum() > 5 and k[mk].std() > 0 else np.nan)
                mu = ref.loc[(d, nt), "mean"]
                pos, neg = m & (g.ks_true > 0), m & (g.ks_true < 0)
                bal = (0.5 * ((v[pos] > mu).mean() + (v[neg] < mu).mean())
                       if pos.any() and neg.any() else np.nan)
                row[f"d{d}"] = f"{r:+.2f}|{rk:+.2f}|{bal:.2f}"
            rows.append(row)
        print(pd.DataFrame(rows).set_index("n_traj").to_string())


if __name__ == "__main__":
    main()
