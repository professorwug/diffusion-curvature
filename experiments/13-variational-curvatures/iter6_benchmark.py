"""Round 6 — estimator innovation on the budget-flat cells.

Arms (each = full pipeline at nt=80):
  r_spec — spectral coordinates (SpectralSF eigenfunctions, lam-weighted)
           feeding td4's data-anchored readout: the "different successor
           features" — smoothest-basis denoising for structured corruption
  r_eq   — td4 with per-state EQUALIZING jitter (homogenize sigma(x))
  r_win  — td4 on temporal WINDOW inputs (x_{t-2..t+2}): implicit drift
           filtering for ar1

Cells: budget-flat {ar1 d4/5/6, hetero d5/6, hd64 d4/6} + sanity
{iso05 d4, clean d4}. 9 combos x 2 profiles x 2 wseeds = 36 units.
Reference: r_td4 from iter4.

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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from diffusion_curvature.variational import (SpectralSF, TDInfoNCE,
                                             spectral_coords)
from noise_benchmark import PROFILES, pooled_dv, prepare_unit

warnings.filterwarnings("ignore")

CELLS = [("ar1", 4), ("ar1", 5), ("ar1", 6), ("hetero", 5), ("hetero", 6),
         ("hd64", 4), ("hd64", 6), ("iso05", 4), ("clean", 4)]
OUT_TPL = "processed_data/iter6_w{wid}.csv"
COLS = ["profile", "d", "wseed", "noise", "arm", "r", "t"]
TD_KW = dict(z_dim=128, features="coords", hidden=256, n_epochs=300,
             batch_size=4096, n_candidates=511, lr=3e-4, holdout_frac=0.5,
             lags_per_step=16, aug_scale=0.5)
ARMS = ("spec", "eq", "win")


def units6():
    return [dict(profile=prof, d=d, wseed=w, noise=noise, arm=arm)
            for noise, d in CELLS for prof in PROFILES for w in (0, 1)
            for arm in ARMS]


def td_readout(prep, X, device, **extra):
    vals = []
    for ns in (0, 1):
        est = TDInfoNCE(gamma=prep["gamma_c"], device=device, seed=ns,
                        **dict(TD_KW, **extra)).fit(
            prep["traj_idx"], prep["n_pts"], X=X)
        vals.append(pooled_dv(est, prep["groups"], rng=ns))
    return np.nanmean(vals, axis=0)


def window_X(X, traj_idx, w=2):
    n_pts, D = X.shape
    nt, T1 = traj_idx.shape
    Xw = np.empty((n_pts, D * (2 * w + 1)), dtype=np.float32)
    Xr = X[traj_idx]                       # (nt, T1, D)
    for j, off in enumerate(range(-w, w + 1)):
        sl = np.clip(np.arange(T1) + off, 0, T1 - 1)
        Xw[traj_idx.ravel(), j * D:(j + 1) * D] = Xr[:, sl].reshape(-1, D)
    return Xw


def run_unit(unit, caches, device):
    prep = prepare_unit(unit, caches, nt=80)
    kt = prep["kt_w"]
    t0 = time.time()
    if unit["arm"] == "spec":
        spec = SpectralSF(gamma=prep["gamma_c"], k_eig=64,
                          features="coords", hidden=256, n_epochs=200,
                          batch_size=4096, lr=1e-3, aug_scale=0.5,
                          aug_nugget="auto", device=device, seed=0).fit(
            prep["traj_idx"], prep["n_pts"], X=prep["X"])
        Psi = spectral_coords(spec, prep["n_pts"])
        v = td_readout(prep, Psi, device, aug_nugget="auto")
    elif unit["arm"] == "eq":
        v = td_readout(prep, prep["X"], device, aug_nugget="equalize")
    else:  # win
        Xw = window_X(prep["X"], prep["traj_idx"])
        v = td_readout(prep, Xw, device, aug_nugget="auto")
    mm = np.isfinite(v) & np.isfinite(kt)
    row = {c: unit[c] for c in ("profile", "d", "wseed", "noise", "arm")}
    row["r"] = pearsonr(v[mm], kt[mm])[0] if mm.sum() > 8 else np.nan
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
    us = units6()
    if args.cmd == "summarize":
        it = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "iter6_w*.csv"))], ignore_index=True)
        it = it.drop_duplicates(["profile", "d", "wseed", "noise", "arm"],
                                keep="last")
        it.to_csv("processed_data/iter6.csv", index=False)
        i4 = pd.read_csv("processed_data/iter4.csv")
        base = i4.groupby(["noise", "d"])["r_td4"].mean().rename("r_td4")
        piv = it.pivot_table(index=["noise", "d"], columns="arm",
                             values="r").join(base)
        pd.set_option("display.width", 240)
        print(f"{len(it)}/{len(us)} iter6 units")
        print(piv.round(3).to_string())
        return
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["wseed"] == 0
              and ((u["noise"], u["d"], u["arm"]) in
                   [("ar1", 4, "win"), ("hetero", 5, "eq"),
                    ("hd64", 4, "spec")])]
    W = args.num_workers
    mine = [(i, u) for i, u in enumerate(us) if i % W == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        dfd = pd.read_csv(out)
        done = set(zip(dfd.profile, dfd.d, dfd.wseed, dfd.noise, dfd.arm))
    header = out.exists() and out.stat().st_size > 0
    caches = {}
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
    for i, u in mine:
        if (u["profile"], u["d"], u["wseed"], u["noise"],
                u["arm"]) in done:
            continue
        try:
            row = run_unit(u, caches, args.device)
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
