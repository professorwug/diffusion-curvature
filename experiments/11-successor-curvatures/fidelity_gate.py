"""Fidelity gate for FB-kernel distances (Stage 1 of FB-Kernel ORC).

Question: do distances derived from the trained FB kernel
K(x,x') = (F(x)^T B(x') + F(x')^T B(x)) / 2 match data-space geodesics at the
scale the Wasserstein-contraction ORC operates (0.3 x median geodesic radius)?

Grid: {sphere, saddle} x gamma {0.5,0.8,0.9,0.98} x z_dim {16,64} x
n_traj {100,500}. Derivations per trained kernel:
  - potential : PHATE-style, L2 between -log rows of the normalized kernel
  - varadhan  : sqrt(-log K_hat)
  - b_l2      : L2 in B-embedding space (negative control = old assumption)
  - f_l2      : L2 in F-embedding space (negative control)

References: graph geodesics (Dijkstra on kNN graph) for both manifolds;
analytic arc-length additionally on the sphere (plus the graph-vs-analytic
ceiling row).

Gate: local Spearman >= ~0.8 for some (gamma, z, derivation).

Usage:
  pixi run python fidelity_gate.py --worker-id 0 --num-workers 2 --device cuda:0
  pixi run python fidelity_gate.py --summarize
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
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from diffusion_curvature.datasets import rejection_sample_from_saddle, sphere
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories

N_POINTS = 2000
KNN = 10
TRAJ_LEN = 50
GAMMAS = (0.5, 0.8, 0.9, 0.98)
Z_DIMS = (16, 64)
N_TRAJS = (100, 500)
DATASETS = ("sphere", "saddle")
N_ANCHORS = 100
LOCAL_RADIUS_FRACTION = 0.3
N_GLOBAL_PAIRS = 20000
SEED = 7

FB_BASE = dict(hidden_dim=256, n_epochs=600, cosine_lr=True, batch_size=1024)

OUT_TPL = "processed_data/fidelity_gate_w{wid}.csv"
OUT_MERGED = Path("processed_data/fidelity_gate.csv")


# ---------------------------------------------------------------------------
# Data + references
# ---------------------------------------------------------------------------


def build_dataset(name: str, seed: int = SEED):
    """Returns (X, D_refs) where D_refs maps reference-name -> (n, n) matrix."""
    if name == "sphere":
        X, _ = sphere(N_POINTS, d=2, seed=seed)
        X = np.asarray(X, dtype=np.float64)
        D_analytic = np.arccos(np.clip(X @ X.T, -1.0, 1.0))
    elif name == "saddle":
        Xs, _ = rejection_sample_from_saddle(N_POINTS, 2)
        X = np.asarray(Xs, dtype=np.float64)
        D_analytic = None
    else:
        raise ValueError(name)

    geo = kneighbors_graph(X, KNN, mode="distance", include_self=False)
    D_graph = dijkstra(geo, directed=False)
    refs = {"graph": D_graph}
    if D_analytic is not None:
        refs["analytic"] = D_analytic
    return X, refs


# ---------------------------------------------------------------------------
# Kernel-distance derivations
# ---------------------------------------------------------------------------


def kernel_distances(M: np.ndarray, F_emb: np.ndarray, B_emb: np.ndarray):
    """Distance matrices derived from the clipped successor kernel M (n, n)."""
    K = 0.5 * (M + M.T)

    out = {}
    rs = K.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    P = K / rs
    U = -np.log(P.astype(np.float64) + 1e-6)
    out["potential"] = pairwise_distances(U)

    K_hat = K / max(K.max(), 1e-12)
    D_var = np.sqrt(-np.log(K_hat + 1e-6))
    np.fill_diagonal(D_var, 0.0)
    out["varadhan"] = D_var

    out["b_l2"] = pairwise_distances(B_emb)
    out["f_l2"] = pairwise_distances(F_emb)
    return out


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------


def fidelity(D_hat: np.ndarray, D_ref: np.ndarray, rng: np.random.Generator):
    """(local_mean, local_std, global, slope_cv) of D_hat against D_ref.

    slope_cv: coefficient of variation, across anchors, of the per-anchor
    median ratio D_hat/D_ref within the local radius. Rank correlation is
    blind to region-dependent scale inflation, but the W1/d ratio in ORC is
    not — slope_cv is the scale-consistency measure that predicts kappa bias
    wobble across the manifold.
    """
    n = D_ref.shape[0]
    finite_ref = np.isfinite(D_ref)

    anchors = rng.choice(n, size=min(N_ANCHORS, n), replace=False)
    local, slopes = [], []
    for a in anchors:
        ref_row = D_ref[a]
        ok = finite_ref[a] & (np.arange(n) != a)
        radius = LOCAL_RADIUS_FRACTION * np.median(ref_row[ok])
        S = np.where(ok & (ref_row <= radius))[0]
        if len(S) < 10:  # sparse anchor: fall back to 20 nearest
            S = np.argsort(np.where(ok, ref_row, np.inf))[:20]
        rho = spearmanr(D_hat[a, S], ref_row[S]).statistic
        if np.isfinite(rho):
            local.append(rho)
        ratio = D_hat[a, S] / np.maximum(ref_row[S], 1e-12)
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio):
            slopes.append(float(np.median(ratio)))

    ii = rng.integers(0, n, size=N_GLOBAL_PAIRS)
    jj = rng.integers(0, n, size=N_GLOBAL_PAIRS)
    keep = (ii != jj) & finite_ref[ii, jj]
    glob = spearmanr(D_hat[ii[keep], jj[keep]], D_ref[ii[keep], jj[keep]]).statistic

    slopes = np.asarray(slopes)
    slope_cv = float(slopes.std() / slopes.mean()) if len(slopes) else float("nan")
    return float(np.mean(local)), float(np.std(local)), float(glob), slope_cv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args) -> None:
    configs = list(itertools.product(DATASETS, GAMMAS, Z_DIMS, N_TRAJS))
    mine = [c for k, c in enumerate(configs)
            if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out)
        done = set(zip(prev.dataset, prev.gamma, prev.z_dim, prev.n_traj))
    print(f"[w{args.worker_id}] {len(mine)} configs on {args.device} "
          f"({len(done)} done)", flush=True)

    data_cache = {}
    header = out.exists() and out.stat().st_size > 0
    t0 = time.time()

    for prog, (ds, gamma, z, n_traj) in enumerate(mine, 1):
        if (ds, gamma, z, n_traj) in done:
            continue
        if ds not in data_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, refs = build_dataset(ds)
                G = pygsp.graphs.NNGraph(X, k=KNN)
            data_cache[ds] = (X, refs, G)
        X, refs, G = data_cache[ds]
        rng = np.random.default_rng(SEED)

        fb_t = time.time()
        traj_idx = subsample_trajectories(
            G, n_trajectories=n_traj, length=TRAJ_LEN, rng=SEED)
        trainer = FBTrainer(
            obs_dim=X.shape[1], z_dim=z, gamma=gamma, **FB_BASE,
            device=args.device, seed=SEED)
        trainer.fit(X[traj_idx].astype(np.float32))
        M, F_emb, B_emb = compute_successor_measures(
            trainer.F_net, trainer.B_net,
            X.astype(np.float32), X.astype(np.float32), device=args.device)
        fb_el = time.time() - fb_t

        rows = []
        for deriv, D_hat in kernel_distances(M, F_emb, B_emb).items():
            for ref_name, D_ref in refs.items():
                lm, ls, gl, scv = fidelity(D_hat, D_ref, rng)
                rows.append(dict(
                    dataset=ds, gamma=gamma, z_dim=z, n_traj=n_traj,
                    derivation=deriv, reference=ref_name,
                    local_spearman=round(lm, 4), local_std=round(ls, 4),
                    global_spearman=round(gl, 4), slope_cv=round(scv, 4),
                    fb_elapsed_s=round(fb_el, 1),
                ))
        # ceiling row: graph geodesics vs analytic (config-independent but
        # cheap; written once per config for convenience)
        if "analytic" in refs and (ds, gamma, z, n_traj) == mine[0]:
            lm, ls, gl, scv = fidelity(refs["graph"], refs["analytic"], rng)
            rows.append(dict(
                dataset=ds, gamma=np.nan, z_dim=np.nan, n_traj=np.nan,
                derivation="graph_geodesic_ceiling", reference="analytic",
                local_spearman=round(lm, 4), local_std=round(ls, 4),
                global_spearman=round(gl, 4), slope_cv=round(scv, 4),
                fb_elapsed_s=0.0,
            ))

        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not header)
        header = True
        best = max(r["local_spearman"] for r in rows
                   if r["reference"] == "graph")
        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} {ds} γ={gamma} z={z} "
              f"n={n_traj} fb={fb_el:.0f}s best_local={best:+.3f} "
              f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)

    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def summarize() -> None:
    frames = [pd.read_csv(p)
              for p in sorted(Path("processed_data").glob("fidelity_gate_w*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "gamma", "z_dim", "n_traj", "derivation", "reference"],
        keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    g = df[df.reference == "graph"]
    piv = g.pivot_table(index=["dataset", "derivation"],
                        columns=["z_dim", "gamma"], values="local_spearman")
    print("LOCAL Spearman vs graph geodesics (max over n_traj):")
    print(g.groupby(["dataset", "derivation", "z_dim", "gamma"])
          .local_spearman.max().unstack(["z_dim", "gamma"]).round(2).to_string())
    print("\nGLOBAL Spearman vs graph geodesics (max over n_traj):")
    print(g.groupby(["dataset", "derivation", "z_dim", "gamma"])
          .global_spearman.max().unstack(["z_dim", "gamma"]).round(2).to_string())
    print("\nSLOPE CV vs graph geodesics (min over n_traj; lower = scale-consistent):")
    print(g.groupby(["dataset", "derivation", "z_dim", "gamma"])
          .slope_cv.min().unstack(["z_dim", "gamma"]).round(2).to_string())
    ceil = df[df.derivation == "graph_geodesic_ceiling"]
    if len(ceil):
        print("\nCeiling (graph geodesics vs analytic, sphere): "
              f"local={ceil.local_spearman.iloc[0]:.3f} "
              f"global={ceil.global_spearman.iloc[0]:.3f}")
    best = g.sort_values("local_spearman", ascending=False).head(8)
    print("\nTop configs by local Spearman:")
    print(best[["dataset", "derivation", "gamma", "z_dim", "n_traj",
                "local_spearman", "global_spearman", "slope_cv"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args)


if __name__ == "__main__":
    main()
