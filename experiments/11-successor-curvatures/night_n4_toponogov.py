"""N-G — Toponogov/Apollonius midpoint comparison on visited clouds.

Purely geodesic signed test: for pairs (a,b) with graph-geodesic midpoint m
and witnesses c, the Euclidean comparison median is
    dbar(m,c) = sqrt(d_ac^2/2 + d_bc^2/2 - d_ab^2/4)   (exact in flat space)
and delta = (d_g(m,c) - dbar)/d_ab is natively signed:
fat triangles (delta > 0) = positive curvature, thin = negative, 0 = flat.

Screen: triads d in {3,4,6}, iid n=2000 AND visited clouds (nt=100, T=50).

Usage: pixi run python night_n4_toponogov.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import pygsp
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances

from diffusion_curvature.trajectory_utils import subsample_trajectories
from fb_kernel_ablation_d4 import build_Xd

warnings.filterwarnings("ignore")

KNN = 10
N_PAIRS = 12
RADIUS_Q = 0.08
DIMS = (3, 4, 6)
DATASETS = ("plane", "sphere", "saddle")
SEEDS = (1, 2, 3)


def toponogov_score(X: np.ndarray, i0: int, rng) -> float:
    n = X.shape[0]
    D = pairwise_distances(X)
    idx = np.argsort(D, axis=1)[:, 1:KNN + 1]
    rows = np.repeat(np.arange(n), KNN)
    cols = idx.ravel()
    G = sp.csr_matrix((D[rows, cols], (rows, cols)), shape=(n, n))
    d_i0 = dijkstra(G, directed=False, indices=[i0])[0]
    finite = np.isfinite(d_i0)
    r0 = np.quantile(d_i0[finite & (d_i0 > 0)], RADIUS_Q)
    # pairs: a = i0, b at radius ~2*r0 (so midpoints sit ~r0 from anchor)
    band = np.where(finite & (d_i0 > 1.6 * r0) & (d_i0 < 2.4 * r0))[0]
    if len(band) < N_PAIRS:
        return np.nan
    bs = rng.choice(band, N_PAIRS, replace=False)
    d_bs = dijkstra(G, directed=False, indices=bs)
    deltas = []
    for q, b in enumerate(bs):
        da, db = d_i0, d_bs[q]
        dab = da[b]
        if not np.isfinite(dab) or dab <= 0:
            continue
        # geodesic midpoint: minimize |da-db| among near-geodesic points
        on_geo = np.where(np.isfinite(da) & np.isfinite(db)
                          & (da + db < 1.05 * dab))[0]
        if len(on_geo) == 0:
            continue
        m = on_geo[np.argmin(np.abs(da[on_geo] - db[on_geo]))]
        d_m = dijkstra(G, directed=False, indices=[m])[0]
        # witnesses: moderate distance from both endpoints
        wit = np.where(np.isfinite(da) & np.isfinite(db) & np.isfinite(d_m)
                       & (da > 0.5 * dab) & (da < 1.5 * dab)
                       & (db > 0.5 * dab) & (db < 1.5 * dab))[0]
        wit = wit[(wit != m) & (wit != i0) & (wit != b)]
        if len(wit) < 8:
            continue
        wit = rng.choice(wit, min(32, len(wit)), replace=False)
        comp = np.sqrt(np.maximum(
            0.5 * da[wit]**2 + 0.5 * db[wit]**2 - 0.25 * dab**2, 1e-12))
        deltas.append(np.median((d_m[wit] - comp) / dab))
    return float(np.mean(deltas)) if deltas else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="processed_data/night_n4.csv")
    args = ap.parse_args()
    rows = []
    for regime in ("iid", "walks"):
        for d in DIMS:
            for ds in DATASETS:
                for seed in SEEDS:
                    np.random.seed(seed)
                    X, ks = build_Xd(ds, d, seed=seed)
                    X = np.asarray(X, dtype=np.float64)
                    rng = np.random.default_rng(seed)
                    t0 = time.time()
                    try:
                        if regime == "walks":
                            G = pygsp.graphs.NNGraph(X, k=KNN)
                            ti = subsample_trajectories(
                                G, n_trajectories=100, length=50,
                                rng=100 + seed)
                            V = np.unique(ti)
                            X_use = X[V]
                            i0 = int(np.argmin(np.linalg.norm(
                                X_use - X[0], axis=1)))
                        else:
                            X_use, i0 = X, 0
                        sc = toponogov_score(X_use, i0, rng)
                    except Exception as e:
                        print(f"  [err] {regime} d={d} {ds} s={seed}: {e}",
                              flush=True)
                        continue
                    rows.append(dict(regime=regime, dim=d, dataset=ds,
                                     seed=seed, ks_true=ks, topo=sc))
                    print(f"{regime} d={d} {ds:<7} s={seed}: "
                          f"topo={sc:+.4f} ({time.time()-t0:.0f}s)",
                          flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    for regime, g in df.groupby("regime"):
        piv = g.pivot_table(index="dim", columns="dataset", values="topo")
        piv["sph-pl"] = piv.sphere - piv.plane
        piv["pl-sad"] = piv.plane - piv.saddle
        mono = ((piv["sph-pl"] > 0) & (piv["pl-sad"] > 0))
        piv["MONO+"] = np.where(mono, "<<<", "")
        print(f"\n=== toponogov [{regime}] ===")
        print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
