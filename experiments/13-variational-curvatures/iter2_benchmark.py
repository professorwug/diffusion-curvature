"""Iteration round 2 on the full N-suite: resolve the hd64-vs-hetero jitter
conflict and map the winnability ceiling.

Channels per unit:
  r_a       — td_aug + nugget + anneal (the hd64 fix)
  r_b       — td_aug base jitter + POOL=64 readout (bigger pooling)
  r_ceiling — sent_knn on TRUE coordinates (winnability map)

Army pattern | summarize (merges with noise_bench_unified.csv).
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

OUT_TPL = "processed_data/iter2_w{wid}.csv"
TD_KW = dict(z_dim=128, features="coords", hidden=256, n_epochs=300,
             batch_size=4096, n_candidates=511, lr=3e-4, holdout_frac=0.5,
             lags_per_step=16, aug_scale=0.5)


def td_field(prep, groups, device, **extra):
    vals = []
    for ns in (0, 1):
        est = TDInfoNCE(gamma=prep["gamma_c"], device=device, seed=ns,
                        **dict(TD_KW, **extra)).fit(
            prep["traj_idx"], prep["n_pts"], X=prep["X"])
        vals.append(pooled_dv(est, groups, rng=ns))
    return np.nanmean(vals, axis=0)


def run_unit(unit, caches, device):
    prep = prepare_unit(unit, caches)
    kt = prep["kt_w"]

    def rw(v):
        mm = np.isfinite(v) & np.isfinite(kt)
        return pearsonr(v[mm], kt[mm])[0] if mm.sum() > 8 else np.nan

    row = dict(unit)
    t0 = time.time()
    # A: nugget + anneal, standard 32-pool groups
    row["r_a"] = rw(td_field(prep, prep["groups"], device,
                             aug_nugget=True, aug_anneal=True))
    # B: base jitter, 64-pool groups
    tree = cKDTree(prep["X"])
    _, grp64 = tree.query(prep["Xt_ev"], k=64)
    row["r_b"] = rw(td_field(prep, list(grp64), device))
    # ceiling: sent_knn on TRUE coords
    Xt = prep["X_true"]
    rng = np.random.default_rng(20)
    sub = rng.choice(prep["n_pts"], 2500, replace=False)
    Dk = np.linalg.norm(Xt[sub][:, None] - Xt[sub][None], axis=-1)
    Wk = affinity_from_D(Dk, k=10)
    Pk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-30)
    ktree = cKDTree(Xt[sub])
    _, kassign = ktree.query(prep["Xt_true"], k=5)
    H_k = entropy_rows(resolvent_rows(Pk, 0.9))
    row["r_ceiling"] = rw(np.nanmean((-H_k)[kassign], axis=1))
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
                            "iter2_w*.csv"))], ignore_index=True)
        it = it.drop_duplicates(["profile", "d", "wseed", "noise"],
                                keep="last")
        it.to_csv("processed_data/iter2.csv", index=False)
        nb = pd.read_csv("processed_data/noise_bench_unified.csv")
        df = nb.merge(it[["profile", "d", "wseed", "noise", "r_a", "r_b",
                          "r_ceiling"]],
                      on=["profile", "d", "wseed", "noise"], how="left")
        df.to_csv("processed_data/noise_bench_iter2.csv", index=False)
        print(f"{it.shape[0]}/72 iter2 units")
        pd.set_option("display.width", 220)
        print(df.groupby(["noise", "d"])[
            ["r_sent_knn", "r_tdaug", "r_a", "r_b", "r_ceiling"]]
            .mean().round(3))
        return
    us = units()
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["d"] == 3
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
