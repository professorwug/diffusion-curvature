"""Iteration round 3: the assembled config (soft nugget) + the budget axis.

Channels per unit:
  r_td3     — td_aug + SOFT nugget (noise-dominance-scaled floor
              subtraction: hard-nugget behavior at hd64, base behavior at
              hetero), nt=40 (standard budget)
  r_td3_2x  — same at nt=80 (2x transitions)
  r_sent2x  — sent_knn (noisy coords) at nt=80
  r_ceil2x  — sent_knn on TRUE coords at nt=80 (does budget lift the
              ceiling past 0.70 at d=4/5?)

Army pattern | summarize (merges with noise_bench_iter2.csv).
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.variational import TDInfoNCE
from noise_benchmark import (entropy_rows, pooled_dv, prepare_unit,
                             resolvent_rows, units)

warnings.filterwarnings("ignore")

OUT_TPL = "processed_data/iter3_w{wid}.csv"
TD_KW = dict(z_dim=128, features="coords", hidden=256, n_epochs=300,
             batch_size=4096, n_candidates=511, lr=3e-4, holdout_frac=0.5,
             lags_per_step=16, aug_scale=0.5, aug_nugget="soft")


def td_field(prep, device):
    vals = []
    for ns in (0, 1):
        est = TDInfoNCE(gamma=prep["gamma_c"], device=device, seed=ns,
                        **TD_KW).fit(prep["traj_idx"], prep["n_pts"],
                                     X=prep["X"])
        vals.append(pooled_dv(est, prep["groups"], rng=ns))
    return np.nanmean(vals, axis=0)


def sent_field(X, Xt_ev, n_pts):
    rng = np.random.default_rng(20)
    sub = rng.choice(n_pts, 2500, replace=False)
    Dk = np.linalg.norm(X[sub][:, None] - X[sub][None], axis=-1)
    Wk = affinity_from_D(Dk, k=10)
    Pk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-30)
    ktree = cKDTree(X[sub])
    _, kassign = ktree.query(Xt_ev, k=5)
    H_k = entropy_rows(resolvent_rows(Pk, 0.9))
    return np.nanmean((-H_k)[kassign], axis=1)


def run_unit(unit, caches, device, skip_1x=False):
    row = dict(unit)
    t0 = time.time()
    if not skip_1x:
        prep = prepare_unit(unit, caches)
        kt = prep["kt_w"]

        def rw(v, ref=None):
            ref = kt if ref is None else ref
            mm = np.isfinite(v) & np.isfinite(ref)
            return pearsonr(v[mm], ref[mm])[0] if mm.sum() > 8 else np.nan

        row["r_td3"] = rw(td_field(prep, device))
    prep2 = prepare_unit(unit, caches, nt=80)
    kt2 = prep2["kt_w"]

    def rw2(v):
        mm = np.isfinite(v) & np.isfinite(kt2)
        return pearsonr(v[mm], kt2[mm])[0] if mm.sum() > 8 else np.nan

    row["r_td3_2x"] = rw2(td_field(prep2, device))
    row["r_sent2x"] = rw2(sent_field(prep2["X"], prep2["Xt_ev"],
                                     prep2["n_pts"]))
    row["r_ceil2x"] = rw2(sent_field(prep2["X_true"], prep2["Xt_true"],
                                     prep2["n_pts"]))
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
        it = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "iter3_w*.csv"))], ignore_index=True)
        it = it.drop_duplicates(["profile", "d", "wseed", "noise"],
                                keep="last")
        it.to_csv("processed_data/iter3.csv", index=False)
        nb = pd.read_csv("processed_data/noise_bench_iter2.csv")
        df = nb.merge(it[["profile", "d", "wseed", "noise", "r_td3",
                          "r_td3_2x", "r_sent2x", "r_ceil2x"]],
                      on=["profile", "d", "wseed", "noise"], how="left")
        df.to_csv("processed_data/noise_bench_iter3.csv", index=False)
        print(f"{it.shape[0]}/72 iter3 units")
        pd.set_option("display.width", 240)
        print(df.groupby(["noise", "d"])[
            ["r_tdaug", "r_a", "r_td3", "r_td3_2x", "r_sent2x",
             "r_ceiling", "r_ceil2x"]].mean().round(3))
        return
    us = units()
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["d"] == 4
              and u["wseed"] == 0 and u["noise"] in ("hetero", "hd64")]
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
