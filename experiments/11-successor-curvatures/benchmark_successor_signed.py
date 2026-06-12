"""Repaired Successor ORC: successor measures + data-space OT, on a battery subset.

For each instance (colosseum d ∈ {2,4}, noise ∈ {0.01, 0.1}, m < 20 from the
fixed-seed signed battery), trains the v2 FB recipe, then runs the signed
Wasserstein estimator with M = softmax(F^T B / tau) and geodesic ground
distances in the data space. For contrast, the *old* (latent-space) Successor
ORC and Successor Entropy are computed from the same trained nets.

Rows appended to processed_data/successor_signed_metrics.csv.

Usage:
  pixi run python benchmark_successor_signed.py --worker-id 0 --num-workers 2 --device cuda:0
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.successor import SuccessorEntropyCurvature, SuccessorORC
from diffusion_curvature.successor.measures import (
    compute_successor_measures,
    softmax_measure,
)
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
OUT_TPL = "processed_data/successor_signed_w{wid}.csv"
OUT_REF = Path("processed_data/successor_signed_metrics.csv")

FB_KW = dict(z_dim=16, hidden_dim=256, gamma=0.9, n_epochs=600,
             cosine_lr=True, batch_size=1024)
N_TRAJECTORIES, TRAJ_LENGTH, KNN = 500, 50, 10
TAUS = (0.3, 1.0)

_ROW_SCHEMA = [
    "dataset", "instance", "method", "ks_hat", "elapsed_s", "fb_elapsed_s",
    "err", "name", "dim", "codim", "noise", "m", "shape", "ks_true",
]


def _subset(instances):
    keep = []
    for i, inst in enumerate(instances):
        if (inst["dataset"] == "colosseum" and inst["dim"] in (2, 4)
                and inst["noise"] in (0.01, 0.1) and inst["m"] < 20):
            keep.append(i)
    return keep


def _existing_done(p: Path):
    if not p.exists() or p.stat().st_size == 0:
        return set()
    df = pd.read_csv(p, usecols=["dataset", "instance", "method"])
    return set(zip(df["dataset"], df["instance"].astype(int), df["method"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    instances = joblib.load(BATTERY_PATH)
    todo = [i for k, i in enumerate(_subset(instances))
            if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = _existing_done(out) | _existing_done(OUT_REF)
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(todo)} instances on {args.device}", flush=True)

    method_names = (
        [f"successor_signed_orc_tau{t:g}" for t in TAUS]
        + [f"successor_signed_orc_phys_tau{t:g}" for t in TAUS]
        + ["successor_orc_old", "successor_entropy_old"]
    )

    t0 = time.time()
    for prog, i in enumerate(todo, 1):
        inst = instances[i]
        if all(("colosseum", i, m) in done for m in method_names):
            continue
        X = inst["X"].astype(np.float32)
        meta = {k: inst[k] for k in
                ("name", "dim", "codim", "noise", "m", "shape", "ks_true")}

        fb_t = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                G = pygsp.graphs.NNGraph(X, k=KNN)
            traj_idx = subsample_trajectories(
                G, n_trajectories=N_TRAJECTORIES, length=TRAJ_LENGTH,
                rng=args.seed + i)
            trainer = FBTrainer(obs_dim=X.shape[1], **FB_KW,
                                device=args.device, seed=args.seed + i)
            trainer.fit(X[traj_idx])
            M, F_emb, B_emb = compute_successor_measures(
                trainer.F_net, trainer.B_net, X, X, device=args.device)
            fb_err = ""
        except Exception as e:
            M = None
            fb_err = str(e)[:200]
            print(f"  [err] FB[{i}]: {fb_err}", flush=True)
        fb_el = round(time.time() - fb_t, 1)

        rows = []
        if M is not None:
            for tau in TAUS:
                t_start = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        P_meas = softmax_measure(M, tau=tau)
                        est = WassersteinSignedCurvature(
                            knn=KNN, n_pairs=8, seed=0, compute_midpoint=False)
                        est.fit(X=X.astype(np.float64), M=P_meas, idx=[0])
                    rows.append((f"successor_signed_orc_tau{tau:g}",
                                 float(est.orc_[0]), time.time() - t_start, ""))
                    rows.append((f"successor_signed_orc_phys_tau{tau:g}",
                                 float(est.orc_phys_[0]), 0.0, ""))
                except Exception as e:
                    err = str(e)[:200]
                    for nm in (f"successor_signed_orc_tau{tau:g}",
                               f"successor_signed_orc_phys_tau{tau:g}"):
                        rows.append((nm, float("nan"), 0.0, err))

            # contrast: original latent-space methods from the same nets
            for nm, fn in (
                ("successor_orc_old", lambda: SuccessorORC(
                    ground="B", k_neighbors=1, top_n=min(150, M.shape[0] - 1),
                    n_projections=32, n_jobs=4,
                ).fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)),
                ("successor_entropy_old", lambda: SuccessorEntropyCurvature(
                ).fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)),
            ):
                t_start = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        k = np.asarray(fn(), dtype=float)
                    k = k[np.isfinite(k)]
                    val = float(k.mean()) if k.size else float("nan")
                    rows.append((nm, val, time.time() - t_start, ""))
                except Exception as e:
                    rows.append((nm, float("nan"), 0.0, str(e)[:200]))
        else:
            rows = [(nm, float("nan"), 0.0, fb_err) for nm in method_names]

        for nm, val, el, err in rows:
            if ("colosseum", i, nm) in done:
                continue
            rec = {c: "" for c in _ROW_SCHEMA}
            rec.update({"dataset": "colosseum", "instance": i, "method": nm,
                        "ks_hat": val, "elapsed_s": round(el, 2),
                        "fb_elapsed_s": fb_el, "err": err, **meta})
            pd.DataFrame([rec])[_ROW_SCHEMA].to_csv(
                out, mode="a", index=False, header=not header)
            header = True

        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(todo)} inst={i} "
              f"d={meta['dim']} eps={meta['noise']} fb={fb_el}s "
              f"eta={(len(todo)-prog)/max(rate,1e-9):.0f}min", flush=True)

    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
