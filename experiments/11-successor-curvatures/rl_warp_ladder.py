"""RL-warp (user's idea, option b): goal-directed policy geometry as a
curvature amplifier.

Goals = top-q positive-proxy points (lowest local diffusion entropy on the
cak kernel, greedily thinned). V = multi-source geodesic travel time from
goals. Induced chain P_pi = softmax(-beta * dV) over kNN neighbors, absorbing
at goals — exactly the chain tabular Q-learning would converge to; its
absorption distribution IS the trained agent's successor-over-goals.

Readout per anchor: basin-mixture entropy = H(absorption distribution over
goals), plus V(anchor) to control the distance confound.
Pre-registered: saddle > plane > sphere at matched V (negative curvature =
watershed splitting; positive = funnel concentration).

Usage: pixi run python rl_warp_ladder.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances

from fb_kernel_ablation_d4 import build_Xd
from graph_ablation_ruler import cak_W

warnings.filterwarnings("ignore")

KNN = 10
GOAL_Q = 0.03
N_GOALS = 12
MIN_GOAL_SEP_Q = 0.15   # thin goals to >= this quantile of geodesic dists
BETA = 6.0
T_ABS = 400
N_ANCHORS = 40
T_ENT = 4
DIMS = (3, 4, 6)
DATASETS = ("plane", "sphere", "saddle")
SEEDS = (1, 2, 3)


def rl_warp_channels(X: np.ndarray, device: str, rng):
    n = X.shape[0]
    D = pairwise_distances(X)
    idx = np.argsort(D, axis=1)[:, 1:KNN + 1]
    rows = np.repeat(np.arange(n), KNN)
    cols = idx.ravel()
    G = sp.csr_matrix((D[rows, cols], (rows, cols)), shape=(n, n))

    # positive-proxy per node: local diffusion entropy on cak kernel (low=+)
    W_cak, _ = cak_W(D, 640)
    P0 = torch.as_tensor(
        W_cak / np.maximum(W_cak.sum(axis=1, keepdims=True), 1e-30),
        dtype=torch.float32, device=device)
    with torch.no_grad():
        Pt = torch.matrix_power(P0, T_ENT)
        Hn = (-(Pt.clamp_min(1e-12) * Pt.clamp_min(1e-12).log())
              .sum(dim=1)).cpu().numpy()

    # goals: lowest-entropy q, greedily thinned by geodesic separation
    order = np.argsort(Hn)
    cand = order[:max(int(GOAL_Q * n) * 3, 30)]
    d_all = dijkstra(G, directed=False, indices=cand[:1])[0]
    sep = np.quantile(d_all[np.isfinite(d_all)], MIN_GOAL_SEP_Q)
    goals = []
    for c in cand:
        if len(goals) >= N_GOALS:
            break
        if all(D[c, g] > sep * 0.7 for g in goals):
            goals.append(int(c))
    for c in cand:                      # relax separation if short
        if len(goals) >= N_GOALS:
            break
        if c not in goals:
            goals.append(int(c))
    goals = np.asarray(goals[:N_GOALS])

    # V = multi-source geodesic travel time; per-goal fields for basins
    d_goals = dijkstra(G, directed=False, indices=goals)
    V = d_goals.min(axis=0)
    scale = np.nanmedian(V[np.isfinite(V) & (V > 0)])

    # induced chain: softmax(-beta * (V(j)-V(i))/scale) over kNN neighbors
    dV = (V[idx] - V[:, None]) / max(scale, 1e-9)
    logits = -BETA * dV
    logits -= logits.max(axis=1, keepdims=True)
    pw = np.exp(logits)
    pw /= pw.sum(axis=1, keepdims=True)
    P_pi = np.zeros((n, n))
    np.put_along_axis(P_pi, idx, pw, axis=1)
    P_pi[goals] = 0.0
    P_pi[goals, goals] = 1.0          # absorbing at goals

    # absorption distribution from each anchor (power iteration)
    anchors = rng.choice(np.setdiff1d(np.arange(n), goals), N_ANCHORS,
                         replace=False)
    Pp = torch.as_tensor(P_pi, dtype=torch.float32, device=device)
    with torch.no_grad():
        rr = torch.zeros((len(anchors), n), device=device)
        for q, a in enumerate(anchors):
            rr[q, a] = 1.0
        for _ in range(T_ABS):
            rr = rr @ Pp
        absorb = rr[:, goals].cpu().numpy()
    absorb = absorb / np.maximum(absorb.sum(axis=1, keepdims=True), 1e-12)
    mix_ent = -(absorb * np.log(np.clip(absorb, 1e-12, 1))).sum(axis=1)
    return pd.DataFrame(dict(
        anchor=anchors, mix_ent=mix_ent, V=V[anchors],
        absorbed=rr.sum(dim=1).cpu().numpy()[:len(anchors)] if False
        else absorb.sum(axis=1))), len(goals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="processed_data/rl_warp_ladder.csv")
    args = ap.parse_args()
    frames = []
    for d in DIMS:
        for ds in DATASETS:
            for seed in SEEDS:
                np.random.seed(seed)
                X, ks = build_Xd(ds, d, seed=seed)
                X = np.asarray(X[:2000], dtype=np.float64)
                rng = np.random.default_rng(seed)
                t0 = time.time()
                try:
                    df, n_goals = rl_warp_channels(X, args.device, rng)
                except Exception as e:
                    print(f"  [err] d={d} {ds} s={seed}: {e}", flush=True)
                    continue
                df["dim"], df["dataset"], df["seed"], df["ks_true"] = \
                    d, ds, seed, ks
                frames.append(df)
                print(f"d={d} {ds:<7} s={seed}: goals={n_goals} "
                      f"mix_ent={df.mix_ent.mean():.3f}±{df.mix_ent.std():.3f} "
                      f"V_med={df.V.median():.2f} ({time.time()-t0:.0f}s)",
                      flush=True)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    # matched-V comparison: within each (dim, seed), quantile-bin V and
    # compare mix_ent across datasets inside bins
    print("\n=== mix_ent by dataset (V-matched via within-dim V-quartile bins) ===")
    allf["V_bin"] = allf.groupby(["dim", "dataset", "seed"]).V.transform(
        lambda v: pd.qcut(v, 4, labels=False, duplicates="drop"))
    piv = allf.groupby(["dim", "dataset"]).mix_ent.mean().unstack()
    piv["sad-pl"] = piv.saddle - piv.plane
    piv["pl-sph"] = piv.plane - piv.sphere
    mono = (piv["sad-pl"] > 0) & (piv["pl-sph"] > 0)
    piv["MONO"] = np.where(mono, "<<<", "")
    print(piv.round(3).to_string())
    print("\nper V-quartile saddle-plane gap:")
    g = allf.groupby(["dim", "dataset", "V_bin"]).mix_ent.mean().unstack(1)
    print((g.saddle - g.plane).unstack().round(3).to_string())


if __name__ == "__main__":
    main()
