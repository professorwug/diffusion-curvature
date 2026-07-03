"""Successor Ricci ladder: edge-perturbation entropy derivatives, four ways.

The DRC idea (diffusion_curvature/ricci.py): bump edge (i,j) by eps, measure
d[laziness_i + laziness_j]/d eps — a signed Ricci proxy needing one graph and
a derivative (no transport plans, no comparison spaces). Variants:

  drc_euclid        : torch-autograd DRC on the visited-cloud euclidean kNN
                      affinity graph (control — replicates the claimed
                      SadSpheres strength in the sparse trajectory regime)
  drc_kernel        : same, on the learned-metric graph (FB kernel distances)
  resolvent_surgery : successor-native — edge bump propagated through the
                      resolvent analytically (dM = gamma(M_:i M_j: + M_:j M_i:)),
                      numerical dH/d eps of the perturbed measure rows
  kernel_injection  : ablation — inject mass into the measure row directly
                      (no propagation)

Node score at anchor i: mean derivative over its k_edge nearest neighbors.
Sign convention: calibrated empirically (reported raw; the triad tells us
which direction means positive).

Grid: triads (plane/sphere/saddle) x d in {2,4,6} x n_traj in {50,200},
4 vmapped FB seeds for the kernel variants.

Usage:
  pixi run python successor_ricci_ladder.py run --worker-id K --num-workers 2 --device cuda:K
  pixi run python successor_ricci_ladder.py summarize
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
import torch

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories

from benchmark_kmetric_colosseum import affinity_from_D, potential_distances
from fb_kernel_ablation_d4 import build_Xd, relu_rownorm

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
TRAJ_LEN = 50
Z_DIM = 64
N_EPOCHS = 300
SEEDS = (7, 8, 9, 10)
N_ANCHORS = 20
K_EDGE = 4          # perturbed edges per anchor (nearest neighbors)
EPS_FD = 1e-4
DIMS = (2, 4, 6)
NTS = (50, 200)
DATASETS = ("plane", "sphere", "saddle")
GAMMA_DIST, GAMMA_MEAS = 0.98, 0.8
SPREAD_FRACTION = 0.3

OUT_TPL = "processed_data/successor_ricci_w{wid}.csv"
OUT_MERGED = Path("processed_data/successor_ricci.csv")


# ---------------------------------------------------------------------------
# DRC via torch autograd (entropy of P^t rows i,j under edge bump)
# ---------------------------------------------------------------------------


def _auto_t_dense(W: np.ndarray, D_ref: np.ndarray, probes, t_max: int = 16) -> int:
    """Spread-targeted t for the dense affinity graph W (distances D_ref)."""
    P = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30)
    target = SPREAD_FRACTION * float(np.median(D_ref[probes]))
    mu = np.zeros((len(probes), W.shape[0]))
    mu[np.arange(len(probes)), probes] = 1.0
    t = 0
    while t < t_max:
        nxt = mu @ P
        spread = float(np.median((nxt * D_ref[probes]).sum(axis=1)))
        if spread > target and t >= 1:
            break
        mu = nxt
        t += 1
    return max(t, 1)


def drc_edge_grads(W: np.ndarray, t: int, edges: list[tuple[int, int]],
                   device: str) -> np.ndarray:
    """d[H_i + H_j]/d eps for an eps bump of edge (i,j) in affinity W.

    Autograd through the t-step row-normalized diffusion. Returns one
    derivative per edge. Higher entropy response = walks spread more when
    connectivity is added.
    """
    Wt = torch.as_tensor(W, dtype=torch.float32, device=device)
    out = []
    for i, j in edges:
        eps = torch.zeros((), dtype=torch.float32, device=device,
                          requires_grad=True)
        W_e = Wt.clone()
        W_e[i, j] = W_e[i, j] + eps
        W_e[j, i] = W_e[j, i] + eps
        P = W_e / W_e.sum(dim=1, keepdim=True).clamp_min(1e-30)
        rows = torch.zeros((2, W.shape[0]), dtype=torch.float32, device=device)
        rows[0, i] = 1.0
        rows[1, j] = 1.0
        for _ in range(t):
            rows = rows @ P
        p = rows.clamp_min(1e-12)
        H = -(p * p.log()).sum()
        (g,) = torch.autograd.grad(H, eps)
        out.append(float(g))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Successor-native surgeries (finite differences on the kernel)
# ---------------------------------------------------------------------------


def _row_entropy(v: np.ndarray) -> float:
    p = v / max(v.sum(), 1e-30)
    p = p[p > 1e-15]
    return float(-(p * np.log(p)).sum())


def resolvent_surgery_grads(M: np.ndarray, gamma: float,
                            edges: list[tuple[int, int]]) -> np.ndarray:
    """dH_i/d eps with the bump propagated through the resolvent:
    dM = gamma (M[:,i] M[j,:] + M[:,j] M[i,:]); rows i and j perturbed."""
    out = []
    scale = float(M.mean())
    for i, j in edges:
        dMi = gamma * (M[i, i] * M[j, :] + M[i, j] * M[i, :])
        dMj = gamma * (M[j, i] * M[j, :] + M[j, j] * M[i, :])
        h0 = _row_entropy(M[i]) + _row_entropy(M[j])
        h1 = (_row_entropy(np.maximum(M[i] + EPS_FD * scale * dMi, 0.0))
              + _row_entropy(np.maximum(M[j] + EPS_FD * scale * dMj, 0.0)))
        out.append((h1 - h0) / (EPS_FD * scale))
    return np.asarray(out)


def kernel_injection_grads(M: np.ndarray,
                           edges: list[tuple[int, int]]) -> np.ndarray:
    """dH_i/d eps for direct mass injection at (i,j) (no propagation)."""
    out = []
    scale = float(M.mean())
    for i, j in edges:
        h0 = _row_entropy(M[i]) + _row_entropy(M[j])
        Mi = M[i].copy()
        Mi[j] += EPS_FD * scale * M[i].sum()
        Mj = M[j].copy()
        Mj[i] += EPS_FD * scale * M[j].sum()
        h1 = _row_entropy(Mi) + _row_entropy(Mj)
        out.append((h1 - h0) / (EPS_FD * scale))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def anchor_edges(D: np.ndarray, anchors, k_edge: int) -> dict[int, list[int]]:
    order = np.argsort(D[anchors], axis=1)
    return {a: [int(j) for j in order[q, 1:k_edge + 1]]
            for q, a in enumerate(anchors)}


def run_worker(args) -> None:
    units = list(itertools.product(DIMS, DATASETS, NTS))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["dim", "dataset", "n_traj"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units on {args.device}", flush=True)

    t0 = time.time()
    for prog, (d, ds_name, nt) in enumerate(mine, 1):
        if (d, ds_name, nt) in done:
            continue
        t1 = time.time()
        X, ks_true = build_Xd(ds_name, d, seed=7)
        X32 = X.astype(np.float32)
        rng = np.random.default_rng(7)
        G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
        traj_idx = subsample_trajectories(G, n_trajectories=nt,
                                          length=TRAJ_LEN, rng=7)
        traj = X32[traj_idx]
        V = np.unique(traj_idx)
        X_V = np.asarray(X[V], dtype=np.float64)
        nV = len(V)
        anchors = rng.choice(nV, min(N_ANCHORS, nV), replace=False).tolist()

        variants: dict[str, list[float]] = {}

        # --- control: DRC on euclidean visited-cloud graph ---
        from sklearn.metrics import pairwise_distances
        D_euc = pairwise_distances(X_V)
        W_euc = affinity_from_D(D_euc, k=KNN)
        t_euc = _auto_t_dense(W_euc, D_euc, anchors)
        edges_map = anchor_edges(D_euc, anchors, K_EDGE)
        edges = [(a, j) for a, js in edges_map.items() for j in js]
        g = drc_edge_grads(W_euc, t_euc, edges, args.device)
        per_anchor = [np.mean(g[q * K_EDGE:(q + 1) * K_EDGE])
                      for q in range(len(anchors))]
        variants["drc_euclid"] = [float(np.mean(per_anchor))]

        # --- FB ensembles for kernel variants ---
        ens_d = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                                  gamma=GAMMA_DIST, n_epochs=N_EPOCHS,
                                  device=args.device)
        ens_d.fit(traj)
        K_dist = np.maximum(ens_d.raw_kernels(X_V.astype(np.float32)), 0.0)
        ens_m = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                                  gamma=GAMMA_MEAS, n_epochs=N_EPOCHS,
                                  device=args.device)
        ens_m.fit(traj)
        K_meas = np.maximum(ens_m.raw_kernels(X_V.astype(np.float32)), 0.0)

        for s in range(len(SEEDS)):
            D_pot = potential_distances(K_dist[s])
            edges_map = anchor_edges(D_pot, anchors, K_EDGE)
            edges = [(a, j) for a, js in edges_map.items() for j in js]

            W_k = affinity_from_D(D_pot, k=KNN)
            t_k = _auto_t_dense(W_k, D_pot, anchors)
            g = drc_edge_grads(W_k, t_k, edges, args.device)
            variants.setdefault("drc_kernel", []).append(float(np.mean(g)))

            M = relu_rownorm(K_meas[s])
            g = resolvent_surgery_grads(M, GAMMA_MEAS, edges)
            variants.setdefault("resolvent_surgery", []).append(float(np.mean(g)))
            g = kernel_injection_grads(M, edges)
            variants.setdefault("kernel_injection", []).append(float(np.mean(g)))

        rows = []
        for name, vals in variants.items():
            v = np.asarray(vals)
            v = v[np.isfinite(v)]
            rows.append(dict(dim=d, dataset=ds_name, n_traj=nt,
                             ks_true=ks_true, variant=name,
                             score=float(v.mean()) if v.size else np.nan,
                             score_std=float(v.std()) if v.size else np.nan,
                             n=len(v)))
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not header)
        header = True
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} d={d} {ds_name} nt={nt} "
              f"({time.time()-t1:.0f}s): "
              + " ".join(f"{r['variant']}={r['score']:+.3g}" for r in rows)
              + f" eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    frames = [pd.read_csv(p) for p in
              sorted(Path("processed_data").glob("successor_ricci_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dim", "dataset", "n_traj", "variant"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    for v in df.variant.unique():
        print(f"=== {v} ===")
        piv = df[df.variant == v].pivot_table(
            index=["dim", "n_traj"], columns="dataset", values="score")
        piv["sph-pl"] = piv.sphere - piv.plane
        piv["pl-sad"] = piv.plane - piv.saddle
        print(piv.round(4).to_string())
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    v2 = sub.add_parser("run-v2")
    v2.add_argument("--worker-id", type=int, default=0)
    v2.add_argument("--num-workers", type=int, default=2)
    v2.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        run_worker(args)
    elif args.cmd == "run-v2":
        run_v2(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# v2: odd/even decomposition of the edge response (central differences).
# The autograd derivative mixes an even (|K|-like) component with a one-sided
# positive term; the odd part of H(eps) should isolate sign.
# ---------------------------------------------------------------------------


def fd_edge_response(W: np.ndarray, t: int, edges, device: str,
                     rel_eps: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference odd and even parts of [H_i + H_j](eps).

    eps is scaled per edge to rel_eps x the mean incident weight, so removal
    (-eps) never makes weights negative.
    """
    import torch as th
    Wt = th.as_tensor(W, dtype=th.float32, device=device)
    n = W.shape[0]
    odd, even = [], []
    with th.no_grad():
        for i, j in edges:
            base = float(0.5 * (Wt[i].mean() + Wt[j].mean()))
            eps = rel_eps * max(base, 1e-12)
            hs = {}
            for sgn in (+1.0, 0.0, -1.0):
                W_e = Wt.clone()
                W_e[i, j] = th.clamp(W_e[i, j] + sgn * eps, min=0.0)
                W_e[j, i] = th.clamp(W_e[j, i] + sgn * eps, min=0.0)
                P = W_e / W_e.sum(dim=1, keepdim=True).clamp_min(1e-30)
                rows = th.zeros((2, n), dtype=th.float32, device=device)
                rows[0, i] = 1.0
                rows[1, j] = 1.0
                for _ in range(t):
                    rows = rows @ P
                p = rows.clamp_min(1e-12)
                hs[sgn] = float(-(p * p.log()).sum())
            odd.append((hs[1.0] - hs[-1.0]) / (2 * eps))
            even.append((hs[1.0] - 2 * hs[0.0] + hs[-1.0]) / (eps**2))
    return np.asarray(odd), np.asarray(even)


def run_v2(args) -> None:
    """Odd/even edge response for euclid + kernel graphs, same grid."""
    units = list(itertools.product(DIMS, DATASETS, NTS))
    mine = [u for i, u in enumerate(units) if i % args.num_workers == args.worker_id]
    out = Path(f"processed_data/successor_ricci_v2_w{args.worker_id}.csv")
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["dim", "dataset", "n_traj"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] v2: {len(mine)} units on {args.device}", flush=True)

    t0 = time.time()
    for prog, (d, ds_name, nt) in enumerate(mine, 1):
        if (d, ds_name, nt) in done:
            continue
        t1 = time.time()
        X, ks_true = build_Xd(ds_name, d, seed=7)
        X32 = X.astype(np.float32)
        rng = np.random.default_rng(7)
        G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
        traj_idx = subsample_trajectories(G, n_trajectories=nt,
                                          length=TRAJ_LEN, rng=7)
        V = np.unique(traj_idx)
        X_V = np.asarray(X[V], dtype=np.float64)
        anchors = rng.choice(len(V), min(N_ANCHORS, len(V)),
                             replace=False).tolist()

        from sklearn.metrics import pairwise_distances
        variants: dict[str, list[float]] = {}

        D_euc = pairwise_distances(X_V)
        W_euc = affinity_from_D(D_euc, k=KNN)
        t_euc = _auto_t_dense(W_euc, D_euc, anchors)
        edges = [(a, j) for a, js in
                 anchor_edges(D_euc, anchors, K_EDGE).items() for j in js]
        odd, even = fd_edge_response(W_euc, t_euc, edges, args.device)
        variants["euclid_odd"] = [float(np.mean(odd))]
        variants["euclid_even"] = [float(np.mean(even))]

        ens_d = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS[:2],
                                  z_dim=Z_DIM, gamma=GAMMA_DIST,
                                  n_epochs=N_EPOCHS, device=args.device)
        ens_d.fit(X32[traj_idx])
        K_dist = np.maximum(ens_d.raw_kernels(X_V.astype(np.float32)), 0.0)
        for s in range(2):
            D_pot = potential_distances(K_dist[s])
            W_k = affinity_from_D(D_pot, k=KNN)
            t_k = _auto_t_dense(W_k, D_pot, anchors)
            edges = [(a, j) for a, js in
                     anchor_edges(D_pot, anchors, K_EDGE).items() for j in js]
            odd, even = fd_edge_response(W_k, t_k, edges, args.device)
            variants.setdefault("kernel_odd", []).append(float(np.mean(odd)))
            variants.setdefault("kernel_even", []).append(float(np.mean(even)))

        rows = [dict(dim=d, dataset=ds_name, n_traj=nt, ks_true=ks_true,
                     variant=k, score=float(np.mean(v)),
                     score_std=float(np.std(v)), n=len(v))
                for k, v in variants.items()]
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not header)
        header = True
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} d={d} {ds_name} nt={nt} "
              f"({time.time()-t1:.0f}s): "
              + " ".join(f"{r['variant']}={r['score']:+.3g}" for r in rows)
              + f" eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] v2 done in {(time.time()-t0)/60:.1f} min", flush=True)
