"""Round 5 — the budget-scaling law: td4 at nt in {160, 320} on the
shortfall cells (iter4 provides nt=80). Constant training compute per unit:
epochs and lag-harvest scale inversely with nt. Summarize fits
r = a + b*log2(nt) per (noise, d) and solves for the nt that buys r=0.70.

Usage: run --nt 160|320 --worker-id .. | summarize
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from diffusion_curvature.variational import TDInfoNCE
from noise_benchmark import PROFILES, pooled_dv, prepare_unit

warnings.filterwarnings("ignore")

SHORTFALL = [("iso15", d) for d in (3, 4, 5, 6)] + \
            [("hetero", d) for d in (3, 4, 5, 6)] + \
            [("ar1", d) for d in (3, 4, 5, 6)] + \
            [("hd64", d) for d in (3, 4, 5, 6)] + \
            [("clean", d) for d in (4, 5, 6)] + [("iso05", 6)]
OUT_TPL = "processed_data/iter5_w{wid}.csv"
COLS = ["profile", "d", "wseed", "noise", "nt", "r_td5", "cv", "t"]


def units5():
    out = []
    for noise, d in SHORTFALL:
        for prof in PROFILES:
            for wseed in (0, 1):
                out.append(dict(profile=prof, d=d, wseed=wseed,
                                noise=noise))
    return out


def run_unit(unit, nt, caches, device):
    prep = prepare_unit(unit, caches, nt=nt)
    kt = prep["kt_w"]
    epochs = max(40, int(round(300 * 80 / nt)))
    lags = max(4, int(round(16 * 80 / nt)))
    t0 = time.time()
    vals, cvs = [], []
    for ns in (0, 1):
        est = TDInfoNCE(gamma=prep["gamma_c"], z_dim=128, features="coords",
                        hidden=256, n_epochs=epochs, batch_size=4096,
                        n_candidates=511, lr=3e-4, holdout_frac=0.5,
                        lags_per_step=lags, aug_scale=0.5,
                        aug_nugget="auto", device=device, seed=ns).fit(
            prep["traj_idx"], prep["n_pts"], X=prep["X"])
        vals.append(pooled_dv(est, prep["groups"], rng=ns))
        cvs.append(est.cv_)
    v = np.nanmean(vals, axis=0)
    mm = np.isfinite(v) & np.isfinite(kt)
    row = {c: unit.get(c) for c in ("profile", "d", "wseed", "noise")}
    row["nt"] = nt
    row["r_td5"] = (pearsonr(v[mm], kt[mm])[0] if mm.sum() > 8 else np.nan)
    row["cv"] = round(float(np.nanmedian(cvs)), 3)
    row["t"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=1)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--nt", type=int, default=160)
    w.add_argument("--test-run", action="store_true")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "summarize":
        i5 = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "iter5_w*.csv"))], ignore_index=True)
        i5 = i5.drop_duplicates(["profile", "d", "wseed", "noise", "nt"],
                                keep="last")
        i5.to_csv("processed_data/iter5.csv", index=False)
        i4 = pd.read_csv("processed_data/iter4.csv")
        i4 = i4[[c for c in ("profile", "d", "wseed", "noise", "r_td4")]]
        i4 = i4.rename(columns={"r_td4": "r_td5"})
        i4["nt"] = 80
        allr = pd.concat([i4, i5[["profile", "d", "wseed", "noise", "nt",
                                  "r_td5"]]], ignore_index=True)
        g = (allr.groupby(["noise", "d", "nt"])["r_td5"].mean()
             .reset_index())
        pd.set_option("display.width", 240)
        piv = g.pivot_table(index=["noise", "d"], columns="nt",
                            values="r_td5").round(3)
        print(f"{len(i5)} iter5 rows")
        print(piv.to_string())
        # scaling fit: r = a + b*log2(nt); nt needed for 0.70
        print("\nnt needed for r=0.70 (log-linear fit):")
        for (noise, d), grp in g.groupby(["noise", "d"]):
            if (noise, d) not in [tuple(x) for x in SHORTFALL]:
                continue
            grp = grp.dropna()
            if len(grp) < 2:
                continue
            b, a = np.polyfit(np.log2(grp.nt), grp.r_td5, 1)
            if b <= 0.005:
                print(f"  {noise} d={d}: flat/declining (b={b:+.3f}) — "
                      f"budget does not buy 0.70")
                continue
            nt70 = 2 ** ((0.70 - a) / b)
            print(f"  {noise} d={d}: slope {b:+.3f}/doubling -> "
                  f"nt~{nt70:,.0f}" + (" (reached)" if nt70 <= 320 else ""))
        return
    us = units5()
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["d"] == 6
              and u["wseed"] == 0 and u["noise"] == "hd64"]
    W = args.num_workers
    mine = [(i, u) for i, u in enumerate(us) if i % W == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        dfd = pd.read_csv(out)
        done = set(zip(dfd.profile, dfd.d, dfd.wseed, dfd.noise, dfd.nt))
    header = out.exists() and out.stat().st_size > 0
    caches = {}
    print(f"[w{args.worker_id}] {len(mine)} units at nt={args.nt}",
          flush=True)
    for i, u in mine:
        if (u["profile"], u["d"], u["wseed"], u["noise"],
                args.nt) in done:
            continue
        try:
            row = run_unit(u, args.nt, caches, args.device)
        except Exception as e:
            print(f"  [err] unit {i}: {str(e)[:150]}", flush=True)
            continue
        pd.DataFrame([row])[COLS].to_csv(out, mode="a", index=False,
                                         header=not header)
        header = True
        print("  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float)
                              else f"{k}={v}" for k, v in row.items()),
              flush=True)
    print(f"[w{args.worker_id}] done", flush=True)


if __name__ == "__main__":
    main()
