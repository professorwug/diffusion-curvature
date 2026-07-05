"""The user's torus thought-experiment: dense-goal successor warp on the
torus of revolution (= necklace k=1), plus the c-stretched symmetrization
test.

Dense goals = ALL points in the positive-curvature region (V = 0 there;
|grad V| = 1 on the interior; no within-region potential structure).
Warp = successor metric of the tilted chain (v3 machinery).

Readouts:
  R1 directional contraction per r-band: ratio d_warp/D for meridian-ish
     vs parallel-ish near pairs (meridian = |dr| dominant). Prediction:
     meridian contraction on the interior; parallels preserved -> interior
     relatively stretched.
  R2 channels on D_warp: interior should read MORE negative than on D.
  R3 c-stretched torus (arc-length reparam profile): warped
     meridian/parallel global ratio vs unwarped — does c=2 move toward
     the c=1 (round) ratio? (symmetrization)

Usage: pixi run python torus_dense_warp.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))

from diffusion_curvature.menagerie import WarpedProduct
from benchmark_kmetric_colosseum import (affinity_from_D,
                                         knn_distance_graph,
                                         potential_distances)
from graph_ablation_ruler import cak_W

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
BETA = 6.0
GAMMA = 0.97
R_BIG, R_TUBE = 1.0, 0.45


def torus_profile(c: float = 1.0):
    """Torus of revolution with tube z-stretched by c, as an arc-length
    reparametrized periodic profile f(s) = R + r cos(phi(s))."""
    phi = np.linspace(0, 2 * np.pi, 40001)
    ds = R_TUBE * np.sqrt(np.sin(phi) ** 2 + (c * np.cos(phi)) ** 2)
    s = np.concatenate([[0], np.cumsum(0.5 * (ds[1:] + ds[:-1])
                                       * np.diff(phi))])
    L = float(s[-1])

    def f(r):
        r = np.asarray(r, dtype=float) % L
        ph = np.interp(r, s, phi)
        return R_BIG + R_TUBE * np.cos(ph)
    return f, L


def sample_torus(c, n, seed, device):
    f, L = torus_profile(c)
    wp = WarpedProduct(f, L, d=2, periodic=True)
    m = wp.sample(n, rng=seed)
    # d=2 sphere factor = S^1: recover angles from the sample's u via ang
    # matrix is already in D; we need per-point alpha: resample explicitly
    rng = np.random.default_rng(seed)
    # regenerate identically (sample uses rng order: r first, then u)
    pdf = np.maximum(wp.fg, 0) ** (wp.d - 1)
    cdf = np.cumsum(pdf); cdf /= cdf[-1]
    r = np.interp(rng.random(n), cdf, wp.rg)
    u = rng.normal(size=(n, 2)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    alpha = np.arctan2(u[:, 1], u[:, 0])
    assert np.allclose(r, m["r"])
    return wp, m, r, alpha, L


def warp_D(D, ks, device, dense=True, goal_q=0.04, seed=0):
    n = D.shape[0]
    W = affinity_from_D(D, k=KNN)
    P_np = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    if dense:
        goals = np.where(ks > 0)[0]
    else:
        goals = np.argsort(-ks)[:max(int(goal_q * n), 8)]
    V = dijkstra(knn_distance_graph(D, k=KNN), directed=False,
                 indices=goals).min(axis=0)
    V = np.where(np.isfinite(V), V, np.nanmax(V[np.isfinite(V)]))
    scale = max(np.median(V[V > 0]), 1e-9)
    tilt = np.exp(-BETA * (V[None, :] - V[:, None]) / scale)
    Pw = P_np * tilt
    Pw = Pw / np.maximum(Pw.sum(1, keepdims=True), 1e-30)
    Pt = torch.as_tensor(Pw, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(len(ks), dtype=torch.float64, device=device) - GAMMA * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    return potential_distances(0.5 * (M + M.T)), V


def ruler(D, anchors, device):
    W_cak, _ = cak_W(D, max(D.shape[0] // 4, 20))
    P = torch.as_tensor(W_cak / np.maximum(W_cak.sum(1, keepdims=True),
                                           1e-30),
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        rows = torch.zeros((len(anchors), D.shape[0]), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for _ in range(4):
            rows = rows @ P
        pr = rows.clamp_min(1e-12)
        return (-(pr * pr.log()).sum(1)).cpu().numpy()


def directional_report(D, Dw, r, alpha, L, wp, label):
    n = len(r)
    dr = np.abs(r[:, None] - r[None, :])
    dr = np.minimum(dr, L - dr)
    da = np.abs(alpha[:, None] - alpha[None, :])
    da = np.minimum(da, 2 * np.pi - da)
    near = (D > 0) & (D < np.quantile(D[D > 0], 0.01))
    mer_frac = np.where(D > 0, dr / np.maximum(D, 1e-12), 0)
    contr = np.where(D > 0, Dw / np.maximum(D, 1e-12), np.nan)
    # r-bands: interior = f near min (R-r), exterior = f near max
    fvals = wp.f(r)
    interior = fvals < np.quantile(fvals, 0.33)
    exterior = fvals > np.quantile(fvals, 0.67)
    rows = []
    for region, mask in [("interior", interior), ("exterior", exterior)]:
        pm = near & mask[:, None] & mask[None, :]
        mer = pm & (mer_frac > 0.8)
        par = pm & (mer_frac < 0.2)
        rows.append(dict(
            label=label, region=region,
            contr_meridian=float(np.nanmedian(contr[mer])) if mer.sum() > 20
            else np.nan,
            contr_parallel=float(np.nanmedian(contr[par])) if par.sum() > 20
            else np.nan,
            n_mer=int(mer.sum()), n_par=int(par.sum())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    all_rows = []
    for c in (1.0, 2.0):
        for seed in (0, 1):
            wp, m, r, alpha, L = sample_torus(c, N, seed, dev)
            D, ks = m["D"], m["ks_field"]
            s = np.median(D[np.triu_indices_from(D, 1)])
            D = D / s
            ks = ks * s**2
            for mode in ("dense", "sparse"):
                Dw, V = warp_D(D, ks, dev, dense=(mode == "dense"),
                               seed=seed)
                Dw = Dw / np.median(Dw[np.triu_indices_from(Dw, 1)])
                rows = directional_report(D, Dw, r, alpha, L, wp,
                                          f"c={c} s={seed} {mode}")
                all_rows += rows
                # R2: ruler field before/after
                order = np.argsort(ks)
                anchors = order[(np.linspace(0.02, 0.98, 40)
                                 * (N - 1)).astype(int)]
                e0 = ruler(D, anchors, dev)
                ew = ruler(Dw, anchors, dev)
                r0 = pearsonr(-e0, ks[anchors])[0]
                rw = pearsonr(-ew, ks[anchors])[0]
                # interior delta: does interior read more negative?
                fv = wp.f(r[anchors])
                inte = fv < np.quantile(fv, 0.33)
                d_int = float((ew - e0)[inte].mean())
                d_ext = float((ew - e0)[~inte].mean())
                print(f"c={c} s={seed} {mode:<6}: ruler field r "
                      f"{r0:+.2f}->{rw:+.2f} | dH interior={d_int:+.3f} "
                      f"exterior={d_ext:+.3f} (pred: int>ext under dense)",
                      flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv("processed_data/torus_dense_warp.csv", index=False)
    print("\n=== R1 directional contraction (median Dw/D, near pairs) ===")
    print(df.to_string(index=False))
