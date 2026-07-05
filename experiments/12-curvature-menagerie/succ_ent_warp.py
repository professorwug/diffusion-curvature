"""Successor entropy as a warp readout (user's suggestion): read the
policy's occupancy operator directly — no re-kernelization, dodging the
density confound that inverted the sample-based channels (R2).

Per eval point: H0 = entropy of unwarped successor row; Hw = warped
(dense-goal tilt); readouts Hw, dH = Hw - H0, and Dw/D ratio for
comparison. Manifolds: torus (k=1), dumbbell d3, necklace d3.
Orientation: higher Hw / dH = more negative K (wide arrival fans).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))

from diffusion_curvature.menagerie import WarpedProduct, necklace_profile, dumbbell_profile
from benchmark_kmetric_colosseum import (affinity_from_D, knn_distance_graph,
                                         potential_distances)
from torus_dense_warp import torus_profile

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
BETA = 6.0
GAMMA = 0.97
DEV = "cuda:0"


def successor(P_np, device=DEV):
    n = P_np.shape[0]
    Pt = torch.as_tensor(P_np, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
    return np.maximum(M.cpu().numpy(), 0.0)


def row_entropy(M, idx):
    R = M[idx]
    R = R / np.maximum(R.sum(1, keepdims=True), 1e-30)
    R = np.clip(R, 1e-15, 1)
    return -(R * np.log(R)).sum(1)


def run(mani, wp_args, seed):
    if mani == "torus":
        f, L = torus_profile(1.0)
        wp = WarpedProduct(f, L, d=2, periodic=True)
    elif mani == "necklace":
        f, L = necklace_profile(b=0.7)
        wp = WarpedProduct(f, L, d=3, periodic=True)
    else:
        f, L = dumbbell_profile(beta=0.8)
        wp = WarpedProduct(f, L, d=3)
    m = wp.sample(N, rng=seed)
    D, ks = m["D"], m["ks_field"]
    s = np.median(D[np.triu_indices_from(D, 1)])
    D, ks = D / s, ks * s**2

    W = affinity_from_D(D, k=KNN)
    P0 = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    goals = np.where(ks > 0)[0]
    V = dijkstra(knn_distance_graph(D, k=KNN), directed=False,
                 indices=goals).min(axis=0)
    V = np.where(np.isfinite(V), V, np.nanmax(V[np.isfinite(V)]))
    scale = max(np.median(V[V > 0]), 1e-9)
    tilt = np.exp(-BETA * (V[None, :] - V[:, None]) / scale)
    Pw = P0 * tilt
    Pw = Pw / np.maximum(Pw.sum(1, keepdims=True), 1e-30)

    M0, Mw = successor(P0), successor(Pw)
    order = np.argsort(ks)
    anchors = order[(np.linspace(0.02, 0.98, 48) * (N - 1)).astype(int)]
    H0 = row_entropy(M0, anchors)
    Hw = row_entropy(Mw, anchors)
    # metric-ratio field for comparison
    Dw = potential_distances(0.5 * (Mw + Mw.T))
    Dw = Dw / np.median(Dw[np.triu_indices_from(Dw, 1)])
    knn_r = []
    for a in anchors:
        nb = np.argsort(D[a])[1:KNN + 1]
        knn_r.append(np.median(Dw[a, nb] / D[a, nb]))
    ratio = np.log(np.asarray(knn_r))
    kt = ks[anchors]
    return dict(
        r_H0=pearsonr(-H0, kt)[0], r_Hw=pearsonr(-Hw, kt)[0],
        r_dH=pearsonr(-(Hw - H0), kt)[0], r_ratio=pearsonr(-ratio, kt)[0])


rows = []
for mani in ("torus", "dumbbell", "necklace"):
    for seed in (0, 1):
        try:
            res = run(mani, None, seed)
        except Exception as e:
            print(f"[err] {mani} s={seed}: {e}", flush=True)
            continue
        rows.append(dict(mani=mani, seed=seed, **res))
        print(f"{mani:<9} s={seed}: " + " ".join(
            f"{k}={v:+.2f}" for k, v in res.items()), flush=True)
df = pd.DataFrame(rows)
df.to_csv("processed_data/succ_ent_warp.csv", index=False)
print("\nmeans:")
print(df.groupby("mani")[["r_H0", "r_Hw", "r_dH", "r_ratio"]]
      .mean().round(2).to_string())
