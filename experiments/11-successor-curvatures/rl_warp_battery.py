"""RL-warp battery: basin-mixture entropy at the origin, colosseum m<15 d>=3,
iid full pointclouds; flat packs = planes with identical goal protocol.
Sign convention: higher mix_ent = more negative K (watershed splitting).
Usage: run/flatpack/summarize (army pattern)."""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from diffusion_curvature.datasets import plane
from rl_warp_ladder import rl_warp_channels

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
DIMS = (3, 4, 5, 6)
N_REP_PACK = 12
OUT_TPL = "processed_data/rlwarp_bat_w{wid}.csv"
PACK_TPL = "processed_data/rlwarp_flat_w{wid}.csv"


def score_instance(X, device, rng):
    df, n_goals = rl_warp_channels(np.asarray(X, dtype=np.float64), device, rng)
    # origin-local: mean mix_ent of the 40-anchor sample's 8 nearest to origin
    X = np.asarray(X, dtype=np.float64)
    d0 = np.linalg.norm(X[df.anchor.values.astype(int)] - X[0], axis=1)
    near = np.argsort(d0)[:8]
    return dict(mix_ent=float(df.mix_ent.values[near].mean()),
                mix_ent_all=float(df.mix_ent.mean()),
                V_anchor=float(df.V.values[near].mean()), n_goals=n_goals)


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
            units = list(itertools.product(DIMS, range(N_REP_PACK)))
        else:
            instances = joblib.load(BATTERY_PATH)
            units = [(i, inst) for i, inst in enumerate(instances)
                     if inst["dataset"] == "colosseum" and inst["m"] < M_MAX
                     and inst["dim"] in DIMS]
        out = Path(tpl.format(wid=args.worker_id))
        keyc = ["dim", "rep"] if is_pack else ["instance"]
        done = set()
        if out.exists() and out.stat().st_size > 0:
            prev = pd.read_csv(out, usecols=keyc)
            done = set(map(tuple, prev.values)) if is_pack else set(
                prev[keyc[0]])
        header = out.exists() and out.stat().st_size > 0
        mine = [u for k, u in enumerate(units)
                if k % args.num_workers == args.worker_id]
        print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
        for u in mine:
            if is_pack:
                d, rep = u
                if (d, rep) in done:
                    continue
                np.random.seed(50_000 + 100 * d + rep)
                X = np.hstack([plane(2000, dim=d), np.zeros((2000, 1))])
                meta = dict(dim=d, rep=rep)
                rng = np.random.default_rng(rep)
            else:
                i, inst = u
                if i in done:
                    continue
                X = np.asarray(inst["X"], dtype=np.float64)[:2000]
                meta = dict(instance=i, dim=inst["dim"], noise=inst["noise"],
                            ks_true=inst["ks_true"])
                rng = np.random.default_rng(7)
            err, sc = "", {}
            try:
                sc = score_instance(X, args.device, rng)
            except Exception as e:
                err = str(e)[:200]
            pd.DataFrame([dict(**meta, err=err, **sc)]).to_csv(
                out, mode="a", index=False, header=not header)
            header = True
        print(f"[w{args.worker_id}] done", flush=True)
    else:
        from scipy.stats import pearsonr
        df = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob("rlwarp_bat_w*.csv"))],
                       ignore_index=True).drop_duplicates(["instance"],
                                                          keep="last")
        flat = pd.concat([pd.read_csv(p) for p in
                          sorted(Path("processed_data").glob(
                              "rlwarp_flat_w*.csv"))], ignore_index=True)
        df.to_csv("processed_data/rlwarp_battery.csv", index=False)
        ref = flat.groupby("dim").mix_ent.agg(["mean", "std"])
        print("=== -mix_ent (origin-local): Pearson | balanced-sign@flat-zero per dim ===")
        for d in DIMS:
            g = df[df.dim == d]
            v = -pd.to_numeric(g.mix_ent, errors="coerce")
            m = np.isfinite(v)
            if m.sum() > 5 and v[m].std() > 0:
                r = pearsonr(v[m], g.ks_true[m])[0]
                mu = -ref.loc[d, "mean"]
                pos, neg = m & (g.ks_true > 0), m & (g.ks_true < 0)
                bal = (0.5 * ((v[pos] > mu).mean() + (v[neg] < mu).mean())
                       if pos.any() and neg.any() else np.nan)
                print(f"d{d}: pearson={r:+.2f} balsign={bal:.2f}")


if __name__ == "__main__":
    main()
