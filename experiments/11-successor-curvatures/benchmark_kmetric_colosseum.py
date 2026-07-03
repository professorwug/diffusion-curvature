"""Overnight: trajectory-regime signed curvature on the Curvature Colosseum.

For each colosseum instance (fixed signed battery, dims 2-6, noise .01-.2,
m < MANIFOLDS_PER_CELL) and each trajectory budget n_traj in {100, 500}:
random walks are sampled on the instance's kNN graph, and every method sees
ONLY the visited nodes (the trajectory regime). Curvature is evaluated at the
visited node nearest the origin, against ks_true at the origin.

Methods (seed-ensembled via the vmapped EnsembleFBTrainer, 4 seeds):
  kmetric_dorc : NEW — "Diffusion ORC on the learned metric". One FB net
                 (gamma=0.98, k=1): kernel -> potential distances -> kNN
                 distance graph (lazy-Dijkstra path metric) + adaptive
                 affinity on the same distances -> graph-diffusion measures
                 (t='auto') -> W1-contraction ORC. Kernel used only for the
                 metric; measures come from validated graph diffusion.
  fbkernel_orc : two-net FB-Kernel ORC baseline (gamma=0.8 relu measures,
                 gamma=0.98 distances).
  traj_dorc    : Diffusion ORC (t='auto') on the visited pointcloud with the
                 ambient euclidean metric (training-free baseline).

Reference rows for the same manifolds with FULL iid data already exist in
signed_metrics.csv (signed_orc_auto / _t8).

Usage:
  pixi run python benchmark_kmetric_colosseum.py run --worker-id K --num-workers 6 --device cuda:X
  pixi run python benchmark_kmetric_colosseum.py summarize
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
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
OUT_TPL = "processed_data/kmetric_colosseum_w{wid}.csv"
OUT_MERGED = Path("processed_data/kmetric_colosseum.csv")

MANIFOLDS_PER_CELL = 15
N_TRAJS = (100, 500)
TRAJ_LEN = 50
KNN = 10
SEEDS = (7, 8, 9, 10)
Z_DIM = 64
N_EPOCHS = 300            # 300 beats 600 under the cosine schedule (A.5 check)
GAMMA_DIST, GAMMA_MEAS = 0.98, 0.8
N_PAIRS = 8

_ROW_SCHEMA = ["instance", "n_traj", "method", "ks_hat", "ks_hat_std",
               "n_seeds_ok", "coverage", "eval_dist", "elapsed_s", "err",
               "name", "dim", "codim", "noise", "m", "ks_true"]


# ---------------------------------------------------------------------------
# Kernel-metric helpers
# ---------------------------------------------------------------------------


def potential_distances(K: np.ndarray) -> np.ndarray:
    rs = K.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    U = -np.log((K / rs).astype(np.float32) + 1e-6)
    return pairwise_distances(U)


def knn_distance_graph(D: np.ndarray, k: int = KNN) -> sp.csr_matrix:
    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    return sp.csr_matrix((D[rows, cols], (rows, cols)), shape=(n, n))


def affinity_from_D(D: np.ndarray, k: int = KNN, alpha: float = 1.0) -> np.ndarray:
    """Adaptive gaussian affinities directly from a distance matrix
    (mirrors kernels.gaussian_kernel's adaptive branch)."""
    sig = np.partition(D, k, axis=1)[:, k]
    sig[sig <= 0] = sig[sig > 0].min() if (sig > 0).any() else 1.0
    W = 0.5 * (np.exp(-(D**2) / (2 * sig[None, :]**2)) / sig[None, :]
               + np.exp(-(D**2) / (2 * sig[:, None]**2)) / sig[:, None])
    if alpha:
        d = 1.0 / (W.sum(axis=1) ** alpha)
        W = W * d[:, None] * d[None, :]
    W[W < 1e-8] = 0.0
    return W


class _SimpleG:
    def __init__(self, W):
        self.W = W


def _seed_mean(vals: list[float]) -> tuple[float, float, int]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std()), int(v.size)


# ---------------------------------------------------------------------------
# Per-unit evaluation
# ---------------------------------------------------------------------------


def evaluate_unit(inst: dict, n_traj: int, device: str) -> list[dict]:
    X = np.asarray(inst["X"], dtype=np.float64)
    G = pygsp.graphs.NNGraph(X, k=KNN)
    traj_idx = subsample_trajectories(
        G, n_trajectories=n_traj, length=TRAJ_LEN, rng=1000 + n_traj)
    V = np.unique(traj_idx)
    X_V = X[V]
    coverage = len(V) / X.shape[0]
    # evaluation point: visited node nearest the origin (X[0])
    d0 = np.linalg.norm(X_V - X[0], axis=1)
    i0 = int(np.argmin(d0))
    eval_dist = float(d0[i0])
    X32 = X_V.astype(np.float32)
    traj_coords = X[traj_idx].astype(np.float32)

    rows = []

    def record(method, vals, t_start, err=""):
        m, s, n_ok = _seed_mean(vals)
        rows.append(dict(method=method, ks_hat=m, ks_hat_std=s,
                         n_seeds_ok=n_ok, coverage=round(coverage, 3),
                         eval_dist=round(eval_dist, 4),
                         elapsed_s=round(time.time() - t_start, 1), err=err))

    # --- training-free baseline ---
    t0 = time.time()
    try:
        est = WassersteinSignedCurvature(t="auto", knn=KNN, n_pairs=N_PAIRS,
                                         seed=0, compute_midpoint=False)
        est.fit(X=X_V, idx=[i0])
        record("traj_dorc", [float(est.orc_[0])], t0)
    except Exception as e:
        record("traj_dorc", [], t0, err=str(e)[:200])

    # --- FB ensembles ---
    t0 = time.time()
    try:
        ens_d = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                                  gamma=GAMMA_DIST, n_epochs=N_EPOCHS,
                                  device=device)
        ens_d.fit(traj_coords)
        K_dist = np.maximum(ens_d.raw_kernels(X32), 0.0)

        ens_m = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                                  gamma=GAMMA_MEAS, n_epochs=N_EPOCHS,
                                  device=device)
        ens_m.fit(traj_coords)
        K_meas = np.maximum(ens_m.raw_kernels(X32), 0.0)
    except Exception as e:
        for m in ("kmetric_dorc", "fbkernel_orc"):
            record(m, [], t0, err=str(e)[:200])
        return rows

    km_vals, fk_vals = [], []
    km_err = fk_err = ""
    for s in range(len(SEEDS)):
        try:
            D_pot = potential_distances(K_dist[s])
            Gk = knn_distance_graph(D_pot)

            # NEW: diffusion measures on the learned metric
            W = affinity_from_D(D_pot)
            est = WassersteinSignedCurvature(t="auto", n_pairs=N_PAIRS,
                                             seed=0, compute_midpoint=False)
            est.fit(G=_SimpleG(W), D_graph=Gk, idx=[i0])
            km_vals.append(float(est.orc_[0]))
        except Exception as e:
            km_err = str(e)[:200]
        try:
            meas = K_meas[s] + 1e-12
            meas = meas / meas.sum(axis=1, keepdims=True)
            est = WassersteinSignedCurvature(n_pairs=N_PAIRS, seed=0,
                                             compute_midpoint=False)
            est.fit(M=meas, D_graph=Gk, idx=[i0])
            fk_vals.append(float(est.orc_[0]))
        except Exception as e:
            fk_err = str(e)[:200]
    record("kmetric_dorc", km_vals, t0, err=km_err)
    record("fbkernel_orc", fk_vals, t0, err=fk_err)
    return rows


# ---------------------------------------------------------------------------
# Worker / summarize
# ---------------------------------------------------------------------------


def _done_all_shards() -> set:
    done = set()
    for p in Path("processed_data").glob("kmetric_colosseum_*.csv"):
        try:
            prev = pd.read_csv(p, usecols=["instance", "n_traj", "method"])
            done |= set(zip(prev.instance, prev.n_traj, prev.method))
        except Exception:
            pass
    return done


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["m"] < MANIFOLDS_PER_CELL]
    units = [(i, inst, nt) for (i, inst), nt in
             itertools.product(cc, N_TRAJS)]
    mine = [u for k, u in enumerate(units) if k % args.num_workers == args.worker_id]
    if args.reverse:
        mine = mine[::-1]

    out = Path(args.out) if args.out else Path(OUT_TPL.format(wid=args.worker_id))
    done = _done_all_shards()
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}{'R' if args.reverse else ''}] {len(mine)} units "
          f"on {args.device} ({len(done)} rows done)", flush=True)

    t0 = time.time()
    for prog, (i, inst, nt) in enumerate(mine, 1):
        if args.reverse:  # helpers race the forward workers: refresh view
            done = _done_all_shards()
        if all((i, nt, m) in done
               for m in ("traj_dorc", "kmetric_dorc", "fbkernel_orc")):
            continue
        meta = {k: inst[k] for k in ("name", "dim", "codim", "noise", "m", "ks_true")}
        rows = evaluate_unit(inst, nt, args.device)
        for r in rows:
            rec = {c: "" for c in _ROW_SCHEMA}
            rec.update({"instance": i, "n_traj": nt, **r, **meta})
            pd.DataFrame([rec])[_ROW_SCHEMA].to_csv(
                out, mode="a", index=False, header=not header)
            header = True
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} inst={i} d={meta['dim']} "
              f"ε={meta['noise']} nt={nt}: "
              + " ".join(f"{r['method']}={r['ks_hat']:+.3f}" for r in rows)
              + f" eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize(args) -> None:
    from scipy.stats import pearsonr
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("kmetric_colosseum_[wh]*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["instance", "n_traj", "method"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    df["ks_hat"] = pd.to_numeric(df.ks_hat, errors="coerce")
    ok = df[df.ks_hat.notna()]
    print(f"wrote {OUT_MERGED} ({len(df)} rows, {len(ok)} finite)\n")

    def bal(g):
        pos, neg = g[g.ks_true > 0], g[g.ks_true < 0]
        if not len(pos) or not len(neg):
            return np.nan
        return 0.5 * ((pos.ks_hat > 0).mean() + (neg.ks_hat < 0).mean())

    for metric_name, fn in [
        ("pearson", lambda g: pearsonr(g.ks_hat, g.ks_true)[0]
                    if len(g) > 3 and g.ks_hat.std() > 0 else np.nan),
        ("balanced sign", bal),
    ]:
        print(f"=== {metric_name} per (method, n_traj, dim), noise pooled ===")
        rows = []
        for (m, nt, d), g in ok.groupby(["method", "n_traj", "dim"]):
            rows.append(dict(method=m, n_traj=nt, dim=d, val=fn(g)))
        s = pd.DataFrame(rows)
        print(s.pivot_table(index=["method", "n_traj"], columns="dim",
                            values="val").round(2).to_string())
        print()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, required=True)
    w.add_argument("--num-workers", type=int, default=6)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--reverse", action="store_true",
                   help="process this worker's units back-to-front (helper "
                        "mode; refreshes the done-set from all shards)")
    w.add_argument("--out", default=None)
    sub.add_parser("summarize")
    args = p.parse_args()
    if args.cmd == "run":
        run_worker(args)
    else:
        run_summarize(args)


if __name__ == "__main__":
    main()
