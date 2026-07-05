"""Track 2 — ratio-field gauntlet: flat control + proxy-quality sensitivity.

Manifolds: dumbbell d3/d4, necklace d3/d4 (fields), flat torus T2/T3
(control: ratio must sit at ~1 with spread = the false-signal floor).
Goal seedings: true (ks>0), proxy-sent (lowest resolvent entropy, top 40%),
proxy-ruler (lowest cak-ent).

Readout: field Pearson of -log(Dw/D) per seeding; flat-torus ratio spread.
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
from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           necklace_profile, torus_flat)
from benchmark_kmetric_colosseum import (affinity_from_D, knn_distance_graph,
                                         potential_distances)
from graph_ablation_ruler import cak_W

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
BETA = 6.0
GAMMA = 0.97
DEV = "cuda:0"


def build(mani, d, seed):
    if mani == "dumbbell":
        f, L = dumbbell_profile(beta=0.8)
        m = WarpedProduct(f, L, d=d).sample(N, rng=seed)
    elif mani == "necklace":
        f, L = necklace_profile(b=0.7)
        m = WarpedProduct(f, L, d=d, periodic=True).sample(N, rng=seed)
    else:
        m = torus_flat(N, d, rng=seed)
    D, ks = m["D"], m["ks_field"]
    s = np.median(D[np.triu_indices_from(D, 1)])
    return D / s, ks * s**2


def resolvent_H(D, device=DEV):
    n = D.shape[0]
    W = affinity_from_D(D, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
    Mn = np.maximum(M.cpu().numpy(), 0.0)
    R = Mn / np.maximum(Mn.sum(1, keepdims=True), 1e-30)
    R = np.clip(R, 1e-15, 1)
    return -(R * np.log(R)).sum(1), P


def cak_H(D, device=DEV):
    W_cak, _ = cak_W(D, max(D.shape[0] // 4, 20))
    P = torch.as_tensor(W_cak / np.maximum(W_cak.sum(1, keepdims=True),
                                           1e-30),
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        Pt = torch.matrix_power(P, 4)
        pr = Pt.clamp_min(1e-12)
        return (-(pr * pr.log()).sum(1)).cpu().numpy()


def ratio_field(D, goals, P0, device=DEV):
    n = D.shape[0]
    V = dijkstra(knn_distance_graph(D, k=KNN), directed=False,
                 indices=goals).min(axis=0)
    V = np.where(np.isfinite(V), V, np.nanmax(V[np.isfinite(V)]))
    scale = max(np.median(V[V > 0]), 1e-9)
    tilt = np.exp(-BETA * (V[None, :] - V[:, None]) / scale)
    Pw = P0 * tilt
    Pw = Pw / np.maximum(Pw.sum(1, keepdims=True), 1e-30)
    Pt = torch.as_tensor(Pw, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
    Mw = np.maximum(M.cpu().numpy(), 0.0)
    Dw = potential_distances(0.5 * (Mw + Mw.T))
    Dw = Dw / np.median(Dw[np.triu_indices_from(Dw, 1)])
    out = np.empty(n)
    for a in range(n):
        nb = np.argsort(D[a])[1:KNN + 1]
        out[a] = np.median(np.log(np.maximum(Dw[a, nb], 1e-12)
                                  / np.maximum(D[a, nb], 1e-12)))
    return out


rows = []
for mani, d in [("dumbbell", 3), ("dumbbell", 4), ("necklace", 3),
                ("necklace", 4), ("torus_flat", 2), ("torus_flat", 3)]:
    for seed in (0, 1):
        D, ks = build(mani, d, seed)
        H0, P0 = resolvent_H(D)
        Hc = cak_H(D)
        n = len(ks)
        seedings = dict(
            true=np.where(ks > 0)[0] if (ks > 0).any()
            else np.argsort(H0)[:int(0.4 * n)],
            proxy_sent=np.argsort(H0)[:int(0.4 * n)],
            proxy_ruler=np.argsort(Hc)[:int(0.4 * n)])
        for name, goals in seedings.items():
            r = ratio_field(D, goals, P0)
            row = dict(mani=mani, dim=d, seed=seed, seeding=name,
                       ratio_med=float(np.median(r)),
                       ratio_iqr=float(np.subtract(
                           *np.percentile(r, [75, 25]))))
            if np.std(ks) > 0:
                row["field_r"] = float(pearsonr(-r, ks)[0])
                acc = (ks[goals] > 0).mean()
                row["goal_acc"] = float(acc)
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v)
                   for k, v in row.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv("processed_data/ratio_gauntlet.csv", index=False)
print("\n=== summary (means over seeds) ===")
print(df.groupby(["mani", "dim", "seeding"])
      [[c for c in ("field_r", "goal_acc", "ratio_med", "ratio_iqr")
        if c in df.columns]].mean().round(3).to_string())
