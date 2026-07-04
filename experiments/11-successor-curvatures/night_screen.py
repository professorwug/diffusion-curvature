"""N1 — overnight screening of five signed-channel candidates on triads.

Channels (pre-registered sign: higher = more positive K):
  ret_t{2,4,8}   : heat-trace return probability P^t_ii (cak_640 kernel)  [N-B]
  bg_ratio       : -( |B(2r)|/|B(r)| )  (Bishop-Gromov; smaller ratio = +K) [N-D]
  ent_concav     : +(H(t2) - 2 H(t4) + H(t8))-style second difference?  We use
                   c2 = H2 - 2*H4 + H8 with sign so that positive K (growth
                   slowdown at larger t) reads positive.                  [N-E]
  resp_skew      : -skew of per-anchor edge-response distribution        [N-F]
  pair_contract  : -mean log(d_k/d_0) over near cross-walk pairs         [N-C]
                   (contraction = positive K), k=5, walks nt=100 T=50

Screen: triads d in {3,4,6}, n=2000, 3 dataset seeds, 30 anchors; report
per-channel triad means + separations + MONO flag.

Usage: pixi run python night_screen.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import pygsp
import torch
from scipy.stats import skew

from diffusion_curvature.trajectory_utils import subsample_trajectories

from benchmark_kmetric_colosseum import affinity_from_D
from diffusing_fraction_colosseum import edge_fractions
from fb_kernel_ablation_d4 import build_Xd
from graph_ablation_ruler import cak_W

warnings.filterwarnings("ignore")

N_POINTS = 2000
KNN = 10
N_ANCHORS = 30
DIMS = (3, 4, 6)
DATASETS = ("plane", "sphere", "saddle")
SEEDS = (1, 2, 3)
CAK_N = 640
TS = (2, 4, 8)
K_PAIR = 5
NT_PAIR, T_PAIR = 100, 50


def channels(X: np.ndarray, device: str, rng) -> dict[str, float]:
    out: dict[str, float] = {}
    n = X.shape[0]
    anchors = rng.choice(n, N_ANCHORS, replace=False)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)

    # --- N-B heat-trace return + N-E concavity (cak kernel) ---
    W_cak, _ = cak_W(D, CAK_N)
    P = torch.as_tensor(
        W_cak / np.maximum(W_cak.sum(axis=1, keepdims=True), 1e-30),
        dtype=torch.float32, device=device)
    with torch.no_grad():
        rows = torch.zeros((len(anchors), n), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        H = {}
        for step in range(1, max(TS) + 1):
            rows = rows @ P
            if step in TS:
                pr = rows.clamp_min(1e-12)
                H[step] = float((-(pr * pr.log()).sum(dim=1)).mean())
                ret = rows.gather(1, torch.as_tensor(
                    anchors, device=device).unsqueeze(1)).squeeze(1)
                out[f"ret_t{step}"] = float(ret.mean())
    out["ent_concav"] = H[2] - 2 * H[4] + H[8]

    # --- N-D Bishop-Gromov ratio ---
    ratios = []
    for a in anchors:
        r = np.partition(D[a], 20)[20]
        n1 = (D[a] <= r).sum() - 1
        n2 = (D[a] <= 2 * r).sum() - 1
        if n1 > 0:
            ratios.append(n2 / n1)
    out["bg_ratio"] = -float(np.mean(ratios))

    # --- N-F response skewness (adaptive k10 kernel, 32 edges/anchor) ---
    W10 = affinity_from_D(D, k=KNN)
    P10 = torch.as_tensor(
        W10 / np.maximum(W10.sum(axis=1, keepdims=True), 1e-30),
        dtype=torch.float32, device=device)
    order = np.argsort(D[anchors], axis=1)
    sk = []
    for q, a in enumerate(anchors):
        edges = [(int(a), int(j)) for j in order[q, 1:33]]
        # reuse edge_fractions machinery? need raw responses; quick inline FD
        from diffusing_fraction_colosseum import edge_fractions as _
        # raw responses via ricci_flow_ladder batched fn
        from ricci_flow_ladder import batched_edge_response
        r = batched_edge_response(P10, np.asarray(edges), t=8)
        sk.append(skew(r))
    out["resp_skew"] = -float(np.nanmean(sk))

    # --- N-C walk-pair coupling contraction ---
    G = pygsp.graphs.NNGraph(X, k=KNN)
    traj = subsample_trajectories(G, n_trajectories=NT_PAIR, length=T_PAIR,
                                  rng=int(rng.integers(1e6)))
    coords = X[traj]                     # (nt, T+1, dim)
    nt, Tp1 = traj.shape
    # find close cross-walk time pairs: subsample candidate pairs
    flat = coords.reshape(-1, X.shape[1])
    ids = np.arange(len(flat))
    walk_of = np.repeat(np.arange(nt), Tp1)
    t_of = np.tile(np.arange(Tp1), nt)
    valid = t_of < Tp1 - K_PAIR
    cand = ids[valid]
    take = rng.choice(cand, min(4000, len(cand)), replace=False)
    A = flat[take]
    ft = torch.as_tensor(flat, dtype=torch.float32, device=device)
    at = torch.as_tensor(A, dtype=torch.float32, device=device)
    Dp = torch.cdist(at, ft).cpu().numpy()
    r_close = np.quantile(Dp[Dp > 0], 0.001)
    logs = []
    for qi, i in enumerate(take):
        js = np.where((Dp[qi] < r_close) & (Dp[qi] > 0)
                      & (walk_of != walk_of[i]) & valid)[0]
        for j in js[:2]:
            d0 = Dp[qi, j]
            wi, ti = walk_of[i], t_of[i]
            wj, tj = walk_of[j], t_of[j]
            dk = np.linalg.norm(coords[wi, ti + K_PAIR]
                                - coords[wj, tj + K_PAIR])
            if d0 > 0 and dk > 0:
                logs.append(np.log(dk / d0))
    out["pair_contract"] = -float(np.mean(logs)) if len(logs) > 20 else np.nan
    out["pair_n"] = float(len(logs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="processed_data/night_screen.csv")
    args = ap.parse_args()
    rows = []
    for d in DIMS:
        for ds in DATASETS:
            for seed in SEEDS:
                np.random.seed(seed)
                X, ks = build_Xd(ds, d, seed=seed)
                X = np.asarray(X, dtype=np.float64)
                rng = np.random.default_rng(seed)
                t0 = time.time()
                try:
                    ch = channels(X, args.device, rng)
                except Exception as e:
                    print(f"  [err] d={d} {ds} s={seed}: {e}", flush=True)
                    continue
                rows.append(dict(dim=d, dataset=ds, seed=seed, ks_true=ks,
                                 **ch))
                print(f"d={d} {ds:<7} s={seed} ({time.time()-t0:.0f}s): "
                      + " ".join(f"{k}={v:+.4g}" for k, v in ch.items()
                                 if k != "pair_n"), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    metrics = [c for c in df.columns
               if c not in ("dim", "dataset", "seed", "ks_true", "pair_n")]
    for m in metrics:
        piv = df.pivot_table(index="dim", columns="dataset", values=m)
        piv["sph-pl"] = piv.sphere - piv.plane
        piv["pl-sad"] = piv.plane - piv.saddle
        mono = ((piv["sph-pl"] > 0) & (piv["pl-sad"] > 0))
        piv["MONO+"] = np.where(mono, "<<<", "")
        print(f"\n=== {m} ===")
        print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
