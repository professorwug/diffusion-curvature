"""Iteration round 4: d in {3,4,5,6}, nt=80 standard budget, budget-scaled
readouts, auto jitter rule, ar1 decorrelation arm.

96 units. Channels:
  r_td4     — TD-InfoNCE, aug_nugget="auto" (CV detector: structured noise
              -> base jitter, unstructured -> soft nugget), nt=80
  r_td4_dec — + min_lag=4 readout-pair decorrelation (ar1 units only)
  r_sent5k  — sent_knn (noisy coords), 5000-point subsample
  r_ceil5k  — sent_knn (TRUE coords), 5000-point subsample (winnability)

Army pattern | summarize.
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
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.variational import TDInfoNCE
from noise_benchmark import (entropy_rows, pooled_dv, prepare_unit,
                             resolvent_rows, units)

warnings.filterwarnings("ignore")

OUT_TPL = "processed_data/iter4_w{wid}.csv"
NT4 = 80
SUB = 5000
TD_KW = dict(z_dim=128, features="coords", hidden=256, n_epochs=300,
             batch_size=4096, n_candidates=511, lr=3e-4, holdout_frac=0.5,
             lags_per_step=16, aug_scale=0.5, aug_nugget="auto")


def td_field(prep, device, **extra):
    vals, cvs = [], []
    for ns in (0, 1):
        est = TDInfoNCE(gamma=prep["gamma_c"], device=device, seed=ns,
                        **dict(TD_KW, **extra)).fit(
            prep["traj_idx"], prep["n_pts"], X=prep["X"])
        vals.append(pooled_dv(est, prep["groups"], rng=ns))
        cvs.append(est.cv_)
    return np.nanmean(vals, axis=0), float(np.nanmedian(cvs))


def sent_field(X, Xt_ev, n_pts, device):
    rng = np.random.default_rng(20)
    sub = rng.choice(n_pts, SUB, replace=False)
    Dk = cdist(X[sub], X[sub])
    Wk = affinity_from_D(Dk, k=10)
    Pk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-30)
    ktree = cKDTree(X[sub])
    _, kassign = ktree.query(Xt_ev, k=5)
    H_k = entropy_rows(resolvent_rows(Pk, 0.9, device=device))
    return np.nanmean((-H_k)[kassign], axis=1)


def run_unit(unit, caches, device):
    prep = prepare_unit(unit, caches, nt=NT4)
    kt = prep["kt_w"]

    def rw(v):
        mm = np.isfinite(v) & np.isfinite(kt)
        return pearsonr(v[mm], kt[mm])[0] if mm.sum() > 8 else np.nan

    row = dict(unit)
    t0 = time.time()
    v, cv = td_field(prep, device)
    row["r_td4"] = rw(v)
    row["cv"] = round(cv, 3)
    if unit["noise"] == "ar1":
        v_dec, _ = td_field(prep, device, min_lag=4)
        row["r_td4_dec"] = rw(v_dec)
    row["r_sent5k"] = rw(sent_field(prep["X"], prep["Xt_ev"],
                                    prep["n_pts"], device))
    row["r_ceil5k"] = rw(sent_field(prep["X_true"], prep["Xt_true"],
                                    prep["n_pts"], device))
    row["t"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    sub_p = ap.add_subparsers(dest="cmd", required=True)
    w = sub_p.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=1)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--test-run", action="store_true")
    sub_p.add_parser("summarize")
    args = ap.parse_args()
    us = units(dims=(3, 4, 5, 6))
    if args.cmd == "summarize":
        it = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "iter4_w*.csv"))], ignore_index=True)
        it = it.drop_duplicates(["profile", "d", "wseed", "noise"],
                                keep="last")
        it.to_csv("processed_data/iter4.csv", index=False)
        print(f"{it.shape[0]}/{len(us)} iter4 units")
        pd.set_option("display.width", 240)
        print(it.groupby(["noise", "d"])[
            ["r_td4", "r_td4_dec", "r_sent5k", "r_ceil5k"]]
            .mean().round(3))
        print("\nCV medians by noise:",
              it.groupby("noise")["cv"].median().round(2).to_dict())
        return
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["d"] == 6
              and u["wseed"] == 0 and u["noise"] in ("clean", "hetero")]
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
