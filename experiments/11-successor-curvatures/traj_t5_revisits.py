"""T5 — revisit scaling: does transitions-per-state (not coverage) govern
coordinate-free channels?

Grid nt in {10, 40} x T in {50, 400, 1600} (matched-total-steps pairs, e.g.
(40,400) vs (10,1600) at 16k steps: same budget, different revisit profile).
Per cell: coverage AND tps = steps/|V| recorded; channels:

  coordinate      : frac (euclid visited graph), ent_cak_t4, kappa_plus
  coordinate-free : frac_emp, ent_emp_t4 (empirical chain);
                    frac_fb, ent_fb (normalized successor kernel)

Prediction (pre-registered): emp/fb channels improve with tps at fixed
coverage; coordinate channels track coverage and saturate in tps.

Usage:
  pixi run python traj_t5_revisits.py run --worker-id K --num-workers W --device cuda:X
  pixi run python traj_t5_revisits.py summarize
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

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories

from traj_t1_survival import traj_channels
from traj_t4_coordfree import coordfree_channels

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
DIMS = (3, 4, 5, 6)
CELLS = [(10, 50), (10, 400), (10, 1600), (40, 50), (40, 400), (40, 1600)]
KNN = 10
MIN_V = 60
SEEDS = (7, 8, 9, 10)
Z_DIM = 64
N_EPOCHS = 300

OUT_TPL = "processed_data/traj_t5_w{wid}.csv"
OUT_MERGED = Path("processed_data/traj_t5.csv")


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["m"] < M_MAX
          and inst["dim"] in DIMS]
    units = [(i, inst, nt, T) for (i, inst), (nt, T) in
             itertools.product(cc, CELLS)]
    mine = [u for k, u in enumerate(units) if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["instance", "n_traj", "traj_len"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)

    t0 = time.time()
    for prog, (i, inst, nt, T) in enumerate(mine, 1):
        if (i, nt, T) in done:
            continue
        X = np.asarray(inst["X"], dtype=np.float64)
        err = ""
        ch: dict[str, float] = {}
        coverage = tps = np.nan
        try:
            G = pygsp.graphs.NNGraph(X, k=KNN)
            traj_idx = subsample_trajectories(
                G, n_trajectories=nt, length=T, rng=1000 + nt + 7 * T)
            V, traj_local = np.unique(traj_idx, return_inverse=True)
            traj_local = traj_local.reshape(traj_idx.shape)
            nV = len(V)
            coverage = nV / X.shape[0]
            tps = traj_idx.size / nV
            X_V = X[V]
            i0 = int(np.argmin(np.linalg.norm(X_V - X[0], axis=1)))
            if nV >= MIN_V:
                ch.update(traj_channels(X_V, i0, args.device))
                ens_op = EnsembleFBTrainer(
                    obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM, gamma=0.8,
                    n_epochs=N_EPOCHS, device=args.device)
                ens_op.fit(X[traj_idx].astype(np.float32))
                K_op = np.maximum(ens_op.raw_kernels(
                    X_V.astype(np.float32)), 0.0)
                ens_d = EnsembleFBTrainer(
                    obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM, gamma=0.98,
                    n_epochs=N_EPOCHS, device=args.device)
                ens_d.fit(X[traj_idx].astype(np.float32))
                K_dist = np.maximum(ens_d.raw_kernels(
                    X_V.astype(np.float32)), 0.0)
                ch.update(coordfree_channels(traj_local, nV, i0, K_op,
                                             K_dist, args.device))
        except Exception as e:
            err = str(e)[:200]
            print(f"  [err] inst={i} nt={nt} T={T}: {err}", flush=True)
        pd.DataFrame([dict(
            instance=i, n_traj=nt, traj_len=T, err=err, dim=inst["dim"],
            noise=inst["noise"], ks_true=inst["ks_true"],
            coverage=round(coverage, 3) if np.isfinite(coverage) else np.nan,
            tps=round(tps, 2) if np.isfinite(tps) else np.nan,
            **ch)]).to_csv(out, mode="a", index=False, header=not header)
        header = True
        if prog % 5 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("traj_t5_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance", "n_traj", "traj_len"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    print("cell medians: coverage | tps")
    print(df.groupby(["n_traj", "traj_len"])[["coverage", "tps"]]
          .median().round(2).to_string())
    channels = [("frac", -1, "coord"), ("ent_cak_t4", -1, "coord"),
                ("kappa_plus", +1, "coord"),
                ("frac_emp", -1, "free"), ("ent_emp_t4", -1, "free"),
                ("frac_fb", -1, "free"), ("ent_fb", -1, "free")]
    for ch, orient, fam in channels:
        if ch not in df.columns:
            continue
        print(f"\n=== {ch} [{fam}]: Pearson per cell (d>=4 pooled) ===")
        rows = []
        for (nt, T), g in df.groupby(["n_traj", "traj_len"]):
            g = g[g.dim >= 4]
            v = orient * pd.to_numeric(g[ch], errors="coerce")
            m = np.isfinite(v)
            r = (pearsonr(v[m], g.ks_true[m])[0]
                 if m.sum() > 5 and v[m].std() > 0 else np.nan)
            rows.append(dict(n_traj=nt, traj_len=T,
                             coverage=round(g.coverage.median(), 2),
                             tps=round(g.tps.median(), 1),
                             pearson=round(r, 2) if np.isfinite(r) else np.nan))
        print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        run_worker(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()
