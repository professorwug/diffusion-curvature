"""Distance-estimator bakeoff for D-abl-2: given a (possibly corrupted) point
cloud, produce a distance matrix by competing estimators, then score the suite.

Estimators (all consume X, return an n x n distance matrix):
  plain        : cdist -> kNN(10) -> Dijkstra geodesics.
  nugget       : variogram-nugget floor subtracted from D^2 before the graph.
  pca          : local-PCA projection denoise (top-d plane of m=30 NN) -> plain.
  diffusion_t  : classical diffusion distance ||P^t_i-P^t_j||_{2,1/pi}, t in {4,8}.
  heatgeo_t    : HeatGeo-style Varadhan distances (vendored; the pypi package
                 pins sklearn 1.2.2 and conflicts). d=sqrt(max(0,-4t log H_t)),
                 H_t = U exp(-t Lambda) U^T of the sym-normalized graph Laplacian.
  fb_potential : early FB-kernel distance (exp-11 fidelity-gate recipe), subset.

Field r (Pearson) is invariant to the flat reference (affine z-scoring), so the
estimator field-r bakeoff and the raw flat-null hallucination are reference-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
EXP11 = HERE.parent / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))
sys.path.insert(0, str(HERE))

import v3_defect as v3  # noqa: E402
from benchmark_kmetric_colosseum import affinity_from_D  # noqa: E402

KNN = 10


def _knn_dijkstra(D: np.ndarray) -> np.ndarray:
    return v3._graph_geodesic(D, k=KNN)


def est_plain(X: np.ndarray, d: int) -> np.ndarray:
    return _knn_dijkstra(cdist(X, X))


def _nugget(D: np.ndarray) -> float:
    Ds = np.sort(D, axis=1)[:, 1:9] ** 2
    ranks = np.arange(1, 9, dtype=float)
    x = ranks - ranks.mean()
    b = (Ds * x).sum(1) / (x @ x)
    a = Ds.mean(1) - b * ranks.mean()
    return float(max(np.median(a), 0.0))


def est_nugget(X: np.ndarray, d: int) -> np.ndarray:
    D = cdist(X, X)
    c0 = _nugget(D)
    Dc = np.sqrt(np.maximum(D**2 - c0, 0.0))
    np.fill_diagonal(Dc, 0.0)
    return _knn_dijkstra(Dc)


def est_pca(X: np.ndarray, d: int, m: int = 30) -> np.ndarray:
    """Project each point onto the top-d PCA plane of its m nearest neighbours."""
    D0 = cdist(X, X)
    order = np.argsort(D0, axis=1)
    Xd = np.empty_like(X)
    for i in range(len(X)):
        nb = X[order[i, :m]]
        c = nb.mean(0)
        _, _, Vt = np.linalg.svd(nb - c, full_matrices=False)
        V = Vt[:d]
        Xd[i] = c + (X[i] - c) @ V.T @ V
    return _knn_dijkstra(cdist(Xd, Xd))


def est_diffusion(X: np.ndarray, d: int, t: int = 4) -> np.ndarray:
    """Classical diffusion distance from the kNN kernel."""
    D0 = cdist(X, X)
    W = affinity_from_D(D0, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    Pt = np.linalg.matrix_power(P, t)
    pi = W.sum(1) / W.sum()
    Q = Pt / np.sqrt(np.maximum(pi, 1e-30))[None, :]
    return cdist(Q, Q)


def est_heatgeo(X: np.ndarray, d: int, t: float = 2.0) -> np.ndarray:
    """HeatGeo-style Varadhan distances (vendored; package version incompatible)."""
    D0 = cdist(X, X)
    W = affinity_from_D(D0, k=KNN)
    W = 0.5 * (W + W.T)
    deg = W.sum(1)
    dm12 = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(len(X)) - (dm12[:, None] * W * dm12[None, :])
    lam, U = np.linalg.eigh(0.5 * (L + L.T))
    Ht = (U * np.exp(-t * np.clip(lam, 0, None))[None, :]) @ U.T
    Ht = 0.5 * (Ht + Ht.T)
    floor = 1e-10
    Ht = np.clip(Ht, floor, None)
    d_hg = np.sqrt(np.maximum(-4.0 * t * np.log(Ht), 0.0))
    np.fill_diagonal(d_hg, 0.0)
    return 0.5 * (d_hg + d_hg.T)


def est_fb(X: np.ndarray, d: int, seed: int = 0, device: str = "cuda:0") -> np.ndarray:
    """Early FB-kernel distance (exp-11 fidelity-gate recipe): walks on the kNN
    graph, FBTrainer(gamma=0.98, z=64, 300 epochs), potential distances (L2 of
    -log successor-kernel rows), Dijkstra re-metrization over k-smallest."""
    import pygsp
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    from sklearn.metrics import pairwise_distances
    from diffusion_curvature.successor.train import FBTrainer
    from diffusion_curvature.successor.measures import compute_successor_measures
    from diffusion_curvature.trajectory_utils import subsample_trajectories
    n = X.shape[0]
    Xf = X.astype(np.float32)
    G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
    traj_idx = subsample_trajectories(G, n_trajectories=100, length=50, rng=seed)
    tr = FBTrainer(obs_dim=X.shape[1], z_dim=64, gamma=0.98, hidden_dim=256,
                   n_epochs=300, cosine_lr=True, batch_size=1024,
                   device=device, seed=seed)
    tr.fit(Xf[traj_idx])
    M, _, _ = compute_successor_measures(tr.F_net, tr.B_net, Xf, Xf, device=device)
    K = 0.5 * (M + M.T)
    P = K / np.maximum(K.sum(1, keepdims=True), 1e-12)
    Dpot = pairwise_distances(-np.log(P.astype(np.float64) + 1e-6))
    idx = np.argsort(Dpot, axis=1)[:, 1:KNN + 1]
    rows = np.repeat(np.arange(n), KNN)
    g = sp.csr_matrix((Dpot[rows, idx.ravel()], (rows, idx.ravel())), shape=(n, n))
    g = g.maximum(g.T)
    Dg = dijkstra(g, directed=False)
    bad = ~np.isfinite(Dg)
    Dg[bad] = Dpot[bad]
    return 0.5 * (Dg + Dg.T)


ESTIMATORS = {
    "plain": lambda X, d: est_plain(X, d),
    "nugget": lambda X, d: est_nugget(X, d),
    "pca": lambda X, d: est_pca(X, d),
    "diffusion_t4": lambda X, d: est_diffusion(X, d, 4),
    "diffusion_t8": lambda X, d: est_diffusion(X, d, 8),
    "heatgeo_t2": lambda X, d: est_heatgeo(X, d, 2.0),
    "heatgeo_t8": lambda X, d: est_heatgeo(X, d, 8.0),
}


def run_bakeoff():
    """Estimators x corruptions on dumbbell beta=0.45 (d3-6, 2 seeds) + flat
    nulls. Field r (reference-free) per channel + raw flat-null hallucination."""
    import time, pandas as pd
    import dist_ablation as da, composite_suite as cs
    from diffusion_curvature.menagerie import dumbbell_profile, WarpedProduct
    CORRUPT = ["B", "C05", "C17", "D"]
    ests = list(ESTIMATORS)
    PROC = HERE / "processed_data"
    t0 = time.time(); rows = []; flat_rows = []
    for d in (3, 4, 5, 6):
        f, L = dumbbell_profile(beta=da.DUMB_BETA); wp = WarpedProduct(f, L, d=d)
        for seed in (0, 1):
            X, ks = da.sample_embedded(wp, 1400, d, seed=seed)
            e = v3.pick_eval_points(ks, 24, np.random.default_rng(seed))
            # oracle A
            o = wp.sample(1400, rng=seed); eo = v3.pick_eval_points(o["ks_field"], 24, np.random.default_rng(seed))
            Do = o["D"] / np.median(o["D"][~np.eye(1400, dtype=bool)]); frA = cs.extract_raw(Do, eo, d)
            for c in cs.CHANNELS:
                rows.append(dict(d=d, seed=seed, corruption="A_oracle", est="oracle",
                                 chan=c, field_r=v3.pearson(frA[c], o["ks_field"][eo])))
            for corr in CORRUPT:
                Xc = da.corrupt(X, corr, seed)
                for est in ests:
                    try:
                        D = ESTIMATORS[est](Xc, d); D = D / np.median(D[~np.eye(1400, dtype=bool)])
                        fr = cs.extract_raw(D, e, d)
                        for c in cs.CHANNELS:
                            rows.append(dict(d=d, seed=seed, corruption=corr, est=est,
                                             chan=c, field_r=v3.pearson(fr[c], ks[e])))
                    except Exception as ex:
                        print(f"  [err] d{d} {corr} {est}: {str(ex)[:60]}", flush=True)
            print(f"[bakeoff] d{d} s{seed} ({time.time()-t0:.0f}s)", flush=True)
    # flat-null hallucination per estimator x corruption (raw |V4|, d=4)
    for corr in CORRUPT:
        Xf, _ = da.flat_plane(1400, 4, 0); Xc = da.corrupt(Xf, corr, 0)
        for est in ests:
            try:
                D = ESTIMATORS[est](Xc, 4); D = D / np.median(D[~np.eye(1400, dtype=bool)])
                fr = cs.extract_raw(D, np.arange(0, 1400, 60), 4)
                flat_rows.append(dict(corruption=corr, est=est,
                                      v4_flat=float(np.median(np.abs(fr.v4_m60))),
                                      v3_flat=float(np.median(np.abs(fr.v3_defect)))))
            except Exception:
                pass
    df = pd.DataFrame(rows); df.to_csv(PROC / "dist_bakeoff_points.csv", index=False)
    fdf = pd.DataFrame(flat_rows); fdf.to_csv(PROC / "dist_bakeoff_flatnull.csv", index=False)
    # summary: V4 and ent_cak field r, mean over d/seed, per (corruption, est)
    pd.set_option("display.width", 240)
    for chan in ("v4_m60", "ent_cak"):
        piv = df[df.chan == chan].pivot_table(index="est", columns="corruption", values="field_r")
        print(f"\n=== {chan} field r (mean over d3-6, 2 seeds) per estimator x corruption ===")
        print(piv.round(2).to_string())
    print("\n=== flat-null raw |V4_m60| (d=4) per estimator x corruption (hallucination) ===")
    print(fdf.pivot_table(index="est", columns="corruption", values="v4_flat").round(3).to_string())
    print(f"[bakeoff] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run_bakeoff()
