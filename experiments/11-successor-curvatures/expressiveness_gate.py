"""Phase A of the expressiveness sprint: exact-resolvent diagnostics for FB.

For the benchmark manifolds we know the walk operator P exactly, so we can
measure the trained kernel's fit against its ground-truth target — the k-step
resolvent density ratio R = n (1-g_k)(I - g_k P^k)^{-1}, g_k = gamma^k — and
test the spectral-alignment prescription (arXiv:2603.20103): k-step temporal
abstraction low-pass-filters the target so a rank-z kernel can represent it.

Per (dataset, k, gamma-role, z, seed):
  - fit_rel_err : scale-optimal relative Frobenius error of raw F^T B vs R
  - row_kl      : mean row-KL of rownorm(relu(K)) vs rownorm(R)
  - distance fidelity (gamma=0.98 role only): local/global Spearman + slope CV
    of potential+Dijkstra distances vs graph geodesics

Plus a no-training spectral table: effective rank of R and the (z+1)-th
singular-value share per (dataset, k, gamma).

Usage:
  pixi run python expressiveness_gate.py spectral
  pixi run python expressiveness_gate.py run --worker-id K --num-workers 2 --device cuda:K
  pixi run python expressiveness_gate.py summarize
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories

from fb_kernel_measure_ablation import build_X, raw_scores
from fb_kernel_orc_sanity import potential_distances, remetrize
from fidelity_gate import fidelity

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
N_TRAJ, TRAJ_LEN = 500, 50
KS = (1, 2, 4, 8)
GAMMAS = (0.8, 0.98)          # measures role, distances role
Z_DIMS = (64, 128)
SEEDS = (7, 8)
DATASETS = ("sphere", "saddle")
FB_BASE = dict(hidden_dim=256, n_epochs=600, cosine_lr=True, batch_size=1024)

OUT_TPL = "processed_data/expressiveness_gate_w{wid}.csv"
OUT_MERGED = Path("processed_data/expressiveness_gate.csv")
OUT_SPECTRAL = Path("processed_data/expressiveness_spectral.csv")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def walk_operator(X: np.ndarray) -> np.ndarray:
    """Row-normalized affinities of the same graph the trajectory sampler uses."""
    G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
    W = np.asarray(G.W.todense(), dtype=np.float64)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    return W / rs


def resolvent_ratio(P: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """R = n (1 - g_k)(I - g_k P^k)^{-1}, g_k = gamma^k (real horizon fixed)."""
    n = P.shape[0]
    gk = gamma**k
    Pk = np.linalg.matrix_power(P, k)
    return n * (1.0 - gk) * np.linalg.inv(np.eye(n) - gk * Pk)


def strided_trajectories(traj_idx: np.ndarray, k: int) -> np.ndarray:
    """All k phase-offset subsamplings, trimmed to equal length and stacked.

    Keeps the total transition count roughly constant across k, isolating the
    spectral effect of abstraction from data-budget effects.
    """
    if k == 1:
        return traj_idx
    parts = [traj_idx[:, o::k] for o in range(k)]
    L = min(p.shape[1] for p in parts)
    return np.concatenate([p[:, :L] for p in parts], axis=0)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def run_spectral(args) -> None:
    rows = []
    for ds in DATASETS:
        X, _ = build_X(ds, seed=7)
        P = walk_operator(X)
        for k, gamma in itertools.product(KS, GAMMAS):
            R = resolvent_ratio(P, k, gamma)
            sv = np.linalg.svd(R, compute_uv=False)
            p = sv / sv.sum()
            erank = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
            rows.append(dict(
                dataset=ds, k=k, gamma=gamma, erank=round(erank, 1),
                sv_share_z64=round(float(sv[64:].sum() / sv.sum()), 4),
                sv_share_z128=round(float(sv[128:].sum() / sv.sum()), 4),
            ))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT_SPECTRAL, index=False)
    print(f"wrote {OUT_SPECTRAL}")


def run_worker(args) -> None:
    units = list(itertools.product(DATASETS, KS, GAMMAS, Z_DIMS, SEEDS))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out)
        done = set(zip(prev.dataset, prev.k, prev.gamma, prev.z_dim, prev.seed))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units on {args.device}", flush=True)

    cache: dict = {}
    t0 = time.time()
    for prog, (ds, k, gamma, z, seed) in enumerate(mine, 1):
        if (ds, k, gamma, z, seed) in done:
            continue
        if ds not in cache:
            X, _ = build_X(ds, seed=7)
            X32 = np.asarray(X, dtype=np.float32)
            P = walk_operator(X)
            G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
            from scipy.sparse.csgraph import dijkstra
            from sklearn.neighbors import kneighbors_graph
            D_graph = dijkstra(
                kneighbors_graph(X, KNN, mode="distance", include_self=False),
                directed=False)
            cache[ds] = dict(X32=X32, P=P, G=G, D_graph=D_graph, R={})
        c = cache[ds]
        if (k, gamma) not in c["R"]:
            c["R"][(k, gamma)] = resolvent_ratio(c["P"], k, gamma)
        R = c["R"][(k, gamma)]

        traj_idx = subsample_trajectories(
            c["G"], n_trajectories=N_TRAJ, length=TRAJ_LEN, rng=seed)
        traj_k = strided_trajectories(traj_idx, k)
        gk = gamma**k

        fb_t = time.time()
        tr = FBTrainer(obs_dim=c["X32"].shape[1], z_dim=z, gamma=gk, **FB_BASE,
                       device=args.device, seed=seed)
        tr.fit(c["X32"][traj_k])
        K_raw = raw_scores(tr, c["X32"], args.device)
        fb_el = time.time() - fb_t

        # fit metrics vs exact resolvent ratio
        alpha = float((K_raw * R).sum() / max((K_raw * K_raw).sum(), 1e-12))
        fit_rel_err = float(np.linalg.norm(alpha * K_raw - R) / np.linalg.norm(R))
        Kp = np.maximum(K_raw, 0.0) + 1e-12
        Kp /= Kp.sum(axis=1, keepdims=True)
        Rp = R / R.sum(axis=1, keepdims=True)
        row_kl = float(np.mean(np.sum(Rp * (np.log(Rp + 1e-12) - np.log(Kp)), axis=1)))

        row = dict(dataset=ds, k=k, gamma=gamma, z_dim=z, seed=seed,
                   gamma_k=round(gk, 4), alpha=round(alpha, 4),
                   fit_rel_err=round(fit_rel_err, 4), row_kl=round(row_kl, 4),
                   fb_elapsed_s=round(fb_el, 1),
                   local_spearman=np.nan, global_spearman=np.nan,
                   slope_cv=np.nan)

        if gamma == 0.98:  # distances role: also score fidelity
            D_hat = remetrize(potential_distances(np.maximum(K_raw, 0.0)))
            lm, _, gl, scv = fidelity(D_hat, c["D_graph"],
                                      np.random.default_rng(seed))
            row.update(local_spearman=round(lm, 4),
                       global_spearman=round(gl, 4), slope_cv=round(scv, 4))

        pd.DataFrame([row]).to_csv(out, mode="a", index=False, header=not header)
        header = True
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} {ds} k={k} γ={gamma} "
              f"z={z} s={seed}: fit={fit_rel_err:.3f} kl={row_kl:.3f} "
              f"loc={row['local_spearman']} eta={(len(mine)-prog)/max(rate,1e-9):.0f}min",
              flush=True)

    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize(args) -> None:
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("expressiveness_gate_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "k", "gamma", "z_dim", "seed"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    for metric in ("fit_rel_err", "row_kl"):
        print(f"{metric} (mean over seeds):")
        print(df.groupby(["gamma", "dataset", "z_dim", "k"])[metric].mean()
              .unstack("k").round(3).to_string())
        print()
    d98 = df[df.gamma == 0.98]
    print("distance fidelity at γ=0.98 (mean over seeds): local | global | slope_cv")
    g = d98.groupby(["dataset", "z_dim", "k"])[
        ["local_spearman", "global_spearman", "slope_cv"]].mean().round(3)
    print(g.to_string())
    if OUT_SPECTRAL.exists():
        print("\nspectral table:")
        print(pd.read_csv(OUT_SPECTRAL).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spectral")
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = p.parse_args()
    if args.cmd == "spectral":
        run_spectral(args)
    elif args.cmd == "run":
        run_worker(args)
    else:
        run_summarize(args)


if __name__ == "__main__":
    main()
