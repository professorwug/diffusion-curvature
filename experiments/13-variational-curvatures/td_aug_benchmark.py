"""td_aug on the N-suite: the unified design — TD-InfoNCE with input-noise
augmentation at 0.5 x the median gamma-lag displacement (data-driven
smoothness prior; see unified_probe{,2}.py for the scale selection).

Same 72 units via prepare_unit. Army pattern | summarize merges with
noise_bench_full.csv.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from diffusion_curvature.variational import TDInfoNCE
from noise_benchmark import pooled_dv, prepare_unit, units

warnings.filterwarnings("ignore")

NET_SEEDS = (0, 1)
AUG = 0.5
OUT_TPL = "processed_data/tdaug_w{wid}.csv"


def run_unit(unit, caches, device):
    prep = prepare_unit(unit, caches)
    X, kt_w = prep["X"], prep["kt_w"]
    t0 = time.time()
    vals, jits = [], []
    for ns in NET_SEEDS:
        est = TDInfoNCE(gamma=prep["gamma_c"], z_dim=128, features="coords",
                        hidden=256, n_epochs=300, batch_size=4096,
                        n_candidates=511, lr=3e-4, holdout_frac=0.5,
                        lags_per_step=16, aug_scale=AUG,
                        device=device, seed=ns).fit(
            prep["traj_idx"], prep["n_pts"], X=X)
        vals.append(pooled_dv(est, prep["groups"], rng=ns))
        jits.append(est.jitter_)
    v = np.nanmean(vals, axis=0)
    mm = np.isfinite(v) & np.isfinite(kt_w)
    row = dict(unit)
    row["r_tdaug"] = (pearsonr(v[mm], kt_w[mm])[0]
                      if mm.sum() > 8 else np.nan)
    row["jit"] = round(float(np.median(jits)), 4)
    row["t"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=1)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--test-run", action="store_true")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "summarize":
        ta = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "tdaug_w*.csv"))], ignore_index=True)
        ta = ta.drop_duplicates(["profile", "d", "wseed", "noise"],
                                keep="last")
        ta.to_csv("processed_data/tdaug.csv", index=False)
        nb = pd.read_csv("processed_data/noise_bench_full.csv")
        df = nb.merge(ta[["profile", "d", "wseed", "noise", "r_tdaug"]],
                      on=["profile", "d", "wseed", "noise"], how="left")
        df.to_csv("processed_data/noise_bench_unified.csv", index=False)
        print(f"{ta.shape[0]}/72 tdaug units")
        pd.set_option("display.width", 200)
        print(df.groupby(["noise", "d"])[
            ["r_sent_bin", "r_sent_knn", "r_fbsent", "r_td", "r_tdaug"]]
            .mean().round(3))
        return
    us = units()
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["d"] == 3
              and u["wseed"] == 0 and u["noise"] in ("clean", "iso15")]
    W = args.num_workers
    mine = [(i, u) for i, u in enumerate(us) if i % W == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        dfd = pd.read_csv(out)
        done = set(zip(dfd.profile, dfd.d, dfd.wseed, dfd.noise))
    header = out.exists() and out.stat().st_size > 0
    caches = {}
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
    for i, u in mine:
        if (u["profile"], u["d"], u["wseed"], u["noise"]) in done:
            continue
        try:
            row = run_unit(u, caches, args.device)
        except Exception as e:
            print(f"  [err] unit {i}: {str(e)[:150]}", flush=True)
            continue
        pd.DataFrame([row]).to_csv(out, mode="a", index=False,
                                   header=not header)
        header = True
        print("  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float)
                              else f"{k}={v}" for k, v in row.items()),
              flush=True)
    print(f"[w{args.worker_id}] done", flush=True)


if __name__ == "__main__":
    main()
