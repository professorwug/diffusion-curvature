"""T3 — kernel-metric channels below the coverage floor (nt in {10, 25}, d>=3).

Same walks as T1 (identical rng -> identical visited sets and eval anchors:
paired comparison). FB ensemble (4 vmapped seeds, gamma=0.98, z=64, 300 ep)
trained on the walk coordinates; per seed, potential distances D_pot replace
D_euc inside all three channels:

  kappa_km   : Diffusion ORC t=8 with P from affinity(D_pot), lazy-Dijkstra
               geodesics over the D_pot kNN graph
  frac_km    : diffusing-edge fraction with D_pot-nearest edges
  entcak_km  : entropy ruler on cak(D_pot, |V|/4), t in {2,4,8}

Gate (from the plan): a km channel earns adoption only if it beats its
euclidean twin / the T2 composite at matched coverage by >= 2 seed-sigma.

Usage:
  pixi run python traj_t3_kernelmetric.py run --worker-id K --num-workers W --device cuda:X
  pixi run python traj_t3_kernelmetric.py summarize
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
import torch

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from benchmark_kmetric_colosseum import (
    _SimpleG,
    affinity_from_D,
    knn_distance_graph,
    potential_distances,
)
from diffusing_fraction_colosseum import edge_fractions
from graph_ablation_ruler import cak_W

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
NTS = (10, 25)
DIMS = (3, 4, 5, 6)
TRAJ_LEN = 50
KNN = 10
T_ORC = 8
T_FRAC = 16
K_EDGE = 16
N_LOCAL = 5
TS_CAK = (2, 4, 8)
MIN_V = 60
SEEDS = (7, 8, 9, 10)
Z_DIM = 64
N_EPOCHS = 300

OUT_TPL = "processed_data/traj_t3_w{wid}.csv"
OUT_MERGED = Path("processed_data/traj_t3.csv")

CH_KM = ("kappa_km", "frac_km", *[f"entcak_km_t{t}" for t in TS_CAK])


def km_channels(D_pot: np.ndarray, i0: int, device: str) -> dict[str, float]:
    out: dict[str, float] = {}
    nV = D_pot.shape[0]
    k_eff = min(KNN, nV - 2)
    W = affinity_from_D(D_pot, k=k_eff)
    Gk = knn_distance_graph(D_pot, k=k_eff)

    est = WassersteinSignedCurvature(t=T_ORC, n_pairs=8, seed=0,
                                     compute_midpoint=False)
    est.fit(G=_SimpleG(W), D_graph=Gk, idx=[i0])
    out["kappa_km"] = float(est.orc_[0])

    P = torch.as_tensor(W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30),
                        dtype=torch.float32, device=device)
    order0 = np.argsort(D_pot[i0])
    anchors = [i0] + [int(j) for j in order0[1:N_LOCAL]]
    order = np.argsort(D_pot[anchors], axis=1)
    k_edge = min(K_EDGE, nV - 2)
    edges = [(a, int(j)) for q, a in enumerate(anchors)
             for j in order[q, 1:k_edge + 1]]
    fr = edge_fractions(P, edges, (T_FRAC,))
    out["frac_km"] = float(fr[T_FRAC].mean())

    W_cak, _ = cak_W(D_pot, max(nV // 4, 20))
    P_cak = torch.as_tensor(
        W_cak / np.maximum(W_cak.sum(axis=1, keepdims=True), 1e-30),
        dtype=torch.float32, device=device)
    with torch.no_grad():
        rows = torch.zeros((len(anchors), nV), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for step in range(1, max(TS_CAK) + 1):
            rows = rows @ P_cak
            if step in TS_CAK:
                pr = rows.clamp_min(1e-12)
                out[f"entcak_km_t{step}"] = float(
                    (-(pr * pr.log()).sum(dim=1)).mean())
    return out


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["m"] < M_MAX
          and inst["dim"] in DIMS]
    units = [(i, inst, nt) for (i, inst), nt in itertools.product(cc, NTS)]
    mine = [u for k, u in enumerate(units) if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["instance", "n_traj"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)

    t0 = time.time()
    for prog, (i, inst, nt) in enumerate(mine, 1):
        if (i, nt) in done:
            continue
        X = np.asarray(inst["X"], dtype=np.float64)
        err = ""
        stats: dict[str, float] = {}
        coverage = np.nan
        try:
            G = pygsp.graphs.NNGraph(X, k=KNN)
            traj_idx = subsample_trajectories(
                G, n_trajectories=nt, length=TRAJ_LEN, rng=1000 + nt)
            V = np.unique(traj_idx)
            coverage = len(V) / X.shape[0]
            X_V = X[V]
            i0 = int(np.argmin(np.linalg.norm(X_V - X[0], axis=1)))
            if len(V) >= MIN_V:
                ens = EnsembleFBTrainer(
                    obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM, gamma=0.98,
                    n_epochs=N_EPOCHS, device=args.device)
                ens.fit(X[traj_idx].astype(np.float32))
                K_dist = np.maximum(
                    ens.raw_kernels(X_V.astype(np.float32)), 0.0)
                per_seed: dict[str, list[float]] = {}
                for s in range(len(SEEDS)):
                    D_pot = potential_distances(K_dist[s])
                    ch = km_channels(D_pot, i0, args.device)
                    for k, v in ch.items():
                        per_seed.setdefault(k, []).append(v)
                for k, vals in per_seed.items():
                    v = np.asarray(vals)
                    v = v[np.isfinite(v)]
                    stats[k] = float(v.mean()) if v.size else np.nan
                    stats[f"{k}_sd"] = float(v.std()) if v.size else np.nan
        except Exception as e:
            err = str(e)[:200]
            print(f"  [err] inst={i} nt={nt}: {err}", flush=True)
        pd.DataFrame([dict(
            instance=i, n_traj=nt, err=err, dim=inst["dim"],
            noise=inst["noise"], ks_true=inst["ks_true"],
            coverage=round(coverage, 3) if np.isfinite(coverage) else np.nan,
            **stats)]).to_csv(out, mode="a", index=False, header=not header)
        header = True
        if prog % 10 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("traj_t3_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance", "n_traj"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    t1 = pd.read_csv("processed_data/traj_t1.csv")
    t1 = t1[t1.n_traj.isin(NTS)]
    m = df.merge(t1[["instance", "n_traj", "kappa_plus", "frac",
                     "ent_cak_t4", "ent_cak_t8"]],
                 on=["instance", "n_traj"], how="left")
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    pairs = [("kappa_km", "kappa_plus", +1), ("frac_km", "frac", -1),
             ("entcak_km_t4", "ent_cak_t4", -1),
             ("entcak_km_t8", "ent_cak_t8", -1)]
    print("=== kernel-metric vs euclidean twin: Pearson per (n_traj, dim) ===")
    for km, euc, orient in pairs:
        rows = []
        for nt in NTS:
            row = {"n_traj": nt}
            for d in DIMS:
                g = m[(m.n_traj == nt) & (m.dim == d)]
                def _r(col):
                    v = orient * pd.to_numeric(g[col], errors="coerce")
                    msk = np.isfinite(v)
                    return (pearsonr(v[msk], g.ks_true[msk])[0]
                            if msk.sum() > 5 and v[msk].std() > 0 else np.nan)
                row[f"d{d}"] = f"{_r(km):+.2f} vs {_r(euc):+.2f}"
            rows.append(row)
        print(f"--- {km} (km vs euclidean) ---")
        print(pd.DataFrame(rows).set_index("n_traj").to_string())


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
