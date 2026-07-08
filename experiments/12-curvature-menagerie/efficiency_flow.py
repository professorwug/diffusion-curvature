"""Efficiency flow (user's unsupervised all-pairs objective): reweight edge
lengths by shortest-path traffic (pooled over ~n*S pairs — pooling built
into the objective), iterate, and decode sign from the DIRECTION channels
move.

Flow: d_e <- d_e * (1 - eta * clip(traffic_hat, -0.8, 3)); global length
renormalization; traffic recomputed each step (feedback). Homogenizing
prediction (pre-registered): under the flow, H rises where K > 0
(redundancy thinned) and falls where K < 0 (necks reinforced):
corr(dH, ks) > 0 on all curved manifolds; dH ~ 0 on the flat null.

Usage: env-python efficiency_flow.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           necklace_profile, torus_flat)
from torus_dense_warp import torus_profile
from benchmark_kmetric_colosseum import affinity_from_D

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
ETA = 0.25
K_STEPS = 4
N_SRC = 300
GAMMA = 0.97
DEV = "cuda:0"


def build_manifold(mani, seed):
    if mani == "dumbbell3":
        f, L = dumbbell_profile(beta=0.8)
        m = WarpedProduct(f, L, d=3).sample(N, rng=seed)
    elif mani == "dumbbell4":
        f, L = dumbbell_profile(beta=0.8)
        m = WarpedProduct(f, L, d=4).sample(N, rng=seed)
    elif mani == "necklace3":
        f, L = necklace_profile(b=0.7)
        m = WarpedProduct(f, L, d=3, periodic=True).sample(N, rng=seed)
    elif mani == "torus_rev":
        f, L = torus_profile(1.0)
        m = WarpedProduct(f, L, d=2, periodic=True).sample(N, rng=seed)
    else:
        m = torus_flat(N, 3, rng=seed)
    D, ks = m["D"], m["ks_field"]
    s = np.median(D[np.triu_indices_from(D, 1)])
    return D / s, ks * s**2


def skeleton(D):
    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:KNN + 1]
    pairs = set()
    for i in range(n):
        for j in idx[i]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    edges = np.array(sorted(pairs))
    return edges, D[edges[:, 0], edges[:, 1]].astype(np.float64)


def edge_traffic(edges, lengths, n, rng):
    G = sp.csr_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(n, n))
    src = rng.choice(n, N_SRC, replace=False)
    dist, pred = dijkstra(G, directed=False, indices=src,
                          return_predecessors=True)
    eid = {}
    for e, (a, b) in enumerate(edges):
        eid[(int(a), int(b))] = e
    traffic = np.zeros(len(edges))
    for si in range(N_SRC):
        d, p = dist[si], pred[si]
        fin = np.isfinite(d)
        order = np.argsort(-d[fin])
        nodes = np.where(fin)[0][order]
        cnt = np.ones(n)
        for j in nodes:
            par = p[j]
            if par < 0:
                continue
            cnt[par] += cnt[j]
            key = (min(int(par), int(j)), max(int(par), int(j)))
            e = eid.get(key)
            if e is not None:
                traffic[e] += cnt[j]
    return traffic


def geodesics(edges, lengths, n):
    G = sp.csr_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(n, n))
    return dijkstra(G, directed=False)


def sent_H(D, anchors, device=DEV):
    n = D.shape[0]
    W = affinity_from_D(D, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
    Mn = np.maximum(M.cpu().numpy(), 0.0)
    R = Mn[anchors]
    R = R / np.maximum(R.sum(1, keepdims=True), 1e-30)
    R = np.clip(R, 1e-15, 1)
    return -(R * np.log(R)).sum(1)


rows = []
for mani in ("dumbbell3", "dumbbell4", "necklace3", "torus_rev",
             "flat_null"):
    for seed in (0, 1):
        t0 = time.time()
        D, ks = build_manifold(mani, seed)
        n = D.shape[0]
        rng = np.random.default_rng(seed)
        edges, lengths = skeleton(D)
        total0 = lengths.sum()
        order = np.argsort(ks)
        anchors = order[(np.linspace(0.02, 0.98, 40) * (n - 1)).astype(int)]
        D0 = geodesics(edges, lengths, n)
        H_prev = sent_H(D0, anchors)
        H0 = H_prev.copy()
        for k in range(K_STEPS):
            tr = edge_traffic(edges, lengths, n, rng)
            t_hat = np.clip(tr / max(tr.mean(), 1e-12) - 1.0, -0.8, 3.0)
            lengths = lengths * (1.0 - ETA * t_hat)
            lengths = np.maximum(lengths, 1e-6)
            lengths *= total0 / lengths.sum()
        Dk = geodesics(edges, lengths, n)
        Hk = sent_H(Dk, anchors)
        dH = Hk - H0
        kt = ks[anchors]
        if np.std(kt) > 0:
            r_dH = pearsonr(dH, kt)[0]
            r_static = pearsonr(-H0, kt)[0]
        else:
            r_dH = r_static = np.nan
        rows.append(dict(mani=mani, seed=seed, r_dH=r_dH,
                         r_static=r_static,
                         dH_mean=float(dH.mean()),
                         dH_absmean=float(np.abs(dH).mean())))
        print(f"{mani:<10} s={seed}: corr(dH, ks)={r_dH:+.2f} "
              f"(static sent r={r_static:+.2f}) "
              f"dH mean={dH.mean():+.3f} |dH|={np.abs(dH).mean():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv("processed_data/efficiency_flow.csv", index=False)
print("\nmeans:")
print(df.groupby("mani")[["r_dH", "r_static", "dH_absmean"]]
      .mean().round(3).to_string())
