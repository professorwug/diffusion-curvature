"""Iteration ablation ladder toward the 0.70-everywhere goal.

6 representative cells x config ladder:
  base    — td_aug as shipped (aug 0.5, random holdout)
  nug     — + nugget-corrected jitter anchor
  nug_ann — + jitter annealing (2x -> 0.25x cosine)
  nug_cf  — nugget + crossfit (complementary holdout phases across 2 nets)
  full    — nugget + anneal + crossfit
  full_l3 — full + TD(lambda=0.3) real-lag grounding
  lkn     — learned-kernel resolvent: short-horizon jittered VMI critic ->
            landmark operator -> analytic resolvent -> entropy
  ceiling — sent_knn on the TRUE (noise-free) coordinates of the same walks
            (how much of the gap is winnable at all)

Army pattern | summarize.  Output: processed_data/iter_probe_w*.csv
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
from diffusion_curvature.variational import VMI, TDInfoNCE
from noise_benchmark import (entropy_rows, pooled_dv, prepare_unit,
                             resolvent_rows)

warnings.filterwarnings("ignore")

CELLS = [
    dict(profile="nk2", d=3, wseed=0, noise="clean"),
    dict(profile="nk2", d=4, wseed=1, noise="clean"),
    dict(profile="nk3", d=4, wseed=0, noise="iso15"),
    dict(profile="nk2", d=4, wseed=0, noise="hetero"),
    dict(profile="nk3", d=4, wseed=1, noise="ar1"),
    dict(profile="nk2", d=4, wseed=0, noise="hd64"),
]
CONFIGS = ("base", "nug", "nug_ann", "nug_cf", "full", "full_l3",
           "lkn", "ceiling")
OUT_TPL = "processed_data/iter_probe_w{wid}.csv"
TD_KW = dict(z_dim=128, features="coords", hidden=256, n_epochs=300,
             batch_size=4096, n_candidates=511, lr=3e-4, holdout_frac=0.5,
             lags_per_step=16, aug_scale=0.5)


def td_field(prep, device, **extra):
    vals, jits = [], []
    for ns in (0, 1):
        kw = dict(TD_KW, **extra)
        if kw.pop("crossfit", False):
            kw["holdout_phase"] = ns % 2
        est = TDInfoNCE(gamma=prep["gamma_c"], device=device, seed=ns,
                        **kw).fit(prep["traj_idx"], prep["n_pts"],
                                  X=prep["X"])
        vals.append(pooled_dv(est, prep["groups"], rng=ns))
        jits.append(est.jitter_)
    return np.nanmean(vals, axis=0), float(np.median(jits))


def lkn_field(prep, device, unit):
    """Short-horizon jittered critic -> landmark one-step-ish operator ->
    analytic resolvent -> entropy at 5 nearest landmarks."""
    X = prep["X"]
    n_pts = prep["n_pts"]
    d = unit["d"]
    # operator horizon: Geom(0.5) lags (mean 2 steps); resolvent gamma'
    # matched to total horizon t_h = 0.09/d at dt=2e-3 per step
    gamma_op = 0.5
    mean_lag = 1.0 / (1.0 - gamma_op)
    gamma_res = 1 - mean_lag * d * 2e-3 / 0.09
    rows = []
    Nl = 1500
    rng = np.random.default_rng(31 + unit["wseed"])
    lm = rng.choice(n_pts, Nl, replace=False)
    for ns in (0, 1):
        est = VMI(gamma=gamma_op, z_dim=128, features="coords", hidden=256,
                  n_epochs=60, batch_size=4096, lags_per_step=8,
                  holdout_frac=0.1, lr=3e-4, aug_scale=0.5,
                  device=device, seed=ns).fit(prep["traj_idx"], n_pts, X=X)
        with torch.no_grad():
            Lt = torch.as_tensor(lm, dtype=torch.long, device=device)
            F = est.net.score(Lt, Lt)                  # (Nl, Nl)
            P = torch.softmax(F, dim=1).double().cpu().numpy()
        H = entropy_rows(resolvent_rows(P, gamma_res, device=device))
        rows.append(-H)
    field = np.mean(rows, axis=0)
    ltree = cKDTree(X[lm])
    _, nassign = ltree.query(prep["Xt_ev"], k=5)
    return np.nanmean(field[nassign], axis=1)


def ceiling_field(prep):
    """sent_knn on the TRUE coordinates of the same walks."""
    Xt = prep["X_true"]
    rng = np.random.default_rng(20 + 0)
    sub = rng.choice(prep["n_pts"], 2500, replace=False)
    Dk = np.linalg.norm(Xt[sub][:, None] - Xt[sub][None], axis=-1)
    Wk = affinity_from_D(Dk, k=10)
    Pk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-30)
    ktree = cKDTree(Xt[sub])
    _, kassign = ktree.query(prep["Xt_true"], k=5)
    H_k = entropy_rows(resolvent_rows(Pk, 0.9))
    return np.nanmean((-H_k)[kassign], axis=1)


def run_unit(cell_i, config, device, caches):
    unit = CELLS[cell_i]
    prep = prepare_unit(unit, caches)
    kt = prep["kt_w"]
    t0 = time.time()
    jit = np.nan
    if config == "base":
        v, jit = td_field(prep, device)
    elif config == "nug":
        v, jit = td_field(prep, device, aug_nugget=True)
    elif config == "nug_ann":
        v, jit = td_field(prep, device, aug_nugget=True, aug_anneal=True)
    elif config == "nug_cf":
        v, jit = td_field(prep, device, aug_nugget=True, crossfit=True)
    elif config == "full":
        v, jit = td_field(prep, device, aug_nugget=True, aug_anneal=True,
                          crossfit=True)
    elif config == "full_l3":
        v, jit = td_field(prep, device, aug_nugget=True, aug_anneal=True,
                          crossfit=True, lam=0.3)
    elif config == "lkn":
        v = lkn_field(prep, device, unit)
    elif config == "ceiling":
        v = ceiling_field(prep)
    mm = np.isfinite(v) & np.isfinite(kt)
    r = pearsonr(v[mm], kt[mm])[0] if mm.sum() > 8 else np.nan
    return dict(cell=cell_i, config=config, **unit, r=r, jit=jit,
                t=round(time.time() - t0, 1))


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
    us = [(ci, cf) for ci in range(len(CELLS)) for cf in CONFIGS]
    if args.cmd == "summarize":
        df = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "iter_probe_w*.csv"))], ignore_index=True)
        df = df.drop_duplicates(["cell", "config"], keep="last")
        df.to_csv("processed_data/iter_probe.csv", index=False)
        pd.set_option("display.width", 220)
        piv = df.pivot_table(index="config", columns="noise", values="r")
        print(f"{len(df)}/{len(us)} units")
        print(piv.round(3))
        return
    if args.test_run:
        us = [(0, "full"), (5, "nug")]
    W = args.num_workers
    mine = [(i, u) for i, u in enumerate(us) if i % W == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        dfd = pd.read_csv(out)
        done = set(zip(dfd.cell, dfd.config))
    header = out.exists() and out.stat().st_size > 0
    caches = {}
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
    for i, (ci, cf) in mine:
        if (ci, cf) in done:
            continue
        try:
            row = run_unit(ci, cf, args.device, caches)
        except Exception as e:
            print(f"  [err] {ci}/{cf}: {str(e)[:150]}", flush=True)
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
