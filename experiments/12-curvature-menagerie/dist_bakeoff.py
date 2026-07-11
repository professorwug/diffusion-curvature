"""Distance-ESTIMATOR bakeoff (deployment-gauntlet ablation #2, part 2).

Part 1 (dist_ablation.py) fixed the distance estimator to plain
cdist->kNN(10)->Dijkstra and swept the *corruption* branch. Here we fix the
corruption branch and sweep the DISTANCE ESTIMATOR that turns the (possibly
corrupted) point cloud into the D matrix the channel suite consumes.

Grid: dumbbell beta=0.45 (the only constructively embeddable signed warped
product, see dist_ablation) x d in {3,4,5} x 2 seeds, plus flat-plane nulls per
(d, branch). Branches {B embed-clean, C05 ambient noise 0.5x kNN, D hd64 lift}
are reused verbatim from dist_ablation.corrupt.

Estimators (each maps X -> n x n distance matrix):
  plain        : cdist -> kNN(10) -> Dijkstra geodesics (dist_ablation baseline).
  pca_denoise  : project each point onto the top-d PCA plane of its m=30 nearest
                 neighbours in the corrupted ambient space, then the plain chain.
  diffdist_t4  : classical diffusion distance from the kNN adaptive kernel.
  diffdist_t8    P=rownorm(W); pi=degree/total; D_t(i,j)=sqrt(sum_a
                 (P^t[i,a]-P^t[j,a])^2/pi[a]) via matrix powers.
  heatgeo_t    : vendored HeatGeo-style Varadhan distance. L_sym of the kNN
  heatgeo_4t     kernel; full eigendecomposition; H_t = U exp(-t Lam) U^T;
                 d=sqrt(max(0,-4t log H_t)); symmetrized. t chosen so the heat
                 kernel's median row-mass radius ~ 0.3x median ambient distance
                 (the diffusion-spread rule) and 4x that. Vendored because the
                 pypi `heatgeo` package pins sklearn 1.2.2 (incompatible here).
  fb_potential : walks (nt=100, T=50) on the branch's kNN graph -> seed-ensembled
                 FB net (gamma=0.98, seeds 7-10, z_dim=64, 300 epochs) ->
                 per-seed potential distances -> averaged -> kNN(10) Dijkstra.

Downstream per (branch, estimator): the channel suite (extract_raw -> v4_m60,
ent_cak, sent + the composite_suite integrator), z-scored against a flat-plane
reference built through the SAME (branch, estimator) chain. D is normalized to
unit median inside extract_raw (repo convention).

CUDA note: this environment's GPU driver is mismatched (forward-compat error
804), so everything runs on CPU. Set JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
in the launching environment; extract_raw and the FB trainer are given
device='cpu'. FB training is ~3 min/instance on CPU, hence the per-cell
checkpointing below (both `refs` and `run` are resumable).

Usage:
  python dist_bakeoff.py refs [--estimators a,b] [--reps N] [--fb-reps N]
  python dist_bakeoff.py run  [--estimators a,b]
  python dist_bakeoff.py summarize
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))
sys.path.insert(0, str(HERE))

import v3_defect as v3  # noqa: E402
import composite_suite as cs  # noqa: E402
import dist_ablation as da  # noqa: E402
from benchmark_kmetric_colosseum import (affinity_from_D,  # noqa: E402
                                         potential_distances)
from diffusion_curvature.menagerie import (dumbbell_profile,  # noqa: E402
                                           WarpedProduct)

PROC = HERE / "processed_data"
DEVICE = "cpu"                       # CUDA unavailable (driver mismatch)
DUMB_BETA = 0.45
DIMS = [3, 4, 5]
SEEDS = [0, 1]
BRANCHES = ["B", "C05", "D"]
ESTIMATORS = ["plain", "pca_denoise", "diffdist_t4", "diffdist_t8",
              "heatgeo_t", "heatgeo_4t", "fb_potential"]
N = 1400
KNN = 10
N_EVAL = 24
REPORT = ["v4_m60", "ent_cak", "sent"]   # channels called out in the brief

# isolated `distbk_p2_` prefix: the other agent (dist_estimators.py) also writes
# dist_bakeoff_*.csv, so part-2 outputs live under a distinct prefix.
POINTS_CSV = PROC / "distbk_p2_points.csv"
REFS_RAW_CSV = PROC / "distbk_p2_refs_raw.csv"


# ---------------------------------------------------------------------------
# Distance estimators (X -> n x n distance matrix)
# ---------------------------------------------------------------------------

def _graph(D: np.ndarray) -> np.ndarray:
    return v3._graph_geodesic(np.asarray(D, dtype=np.float64), k=KNN)


def est_pca_denoise(X: np.ndarray, d: int, m: int = 30) -> np.ndarray:
    """Local-PCA denoise: each point -> its projection onto the top-d PCA plane
    of its m nearest neighbours (corrupted ambient space), then the plain chain."""
    D0 = cdist(X, X)
    order = np.argsort(D0, axis=1)
    Xd = np.empty_like(X)
    for i in range(len(X)):
        nb = X[order[i, :m]]
        c = nb.mean(0)
        _, _, Vt = np.linalg.svd(nb - c, full_matrices=False)
        V = Vt[:d]
        Xd[i] = c + (X[i] - c) @ V.T @ V
    return _graph(cdist(Xd, Xd))


def est_diffdist(D0: np.ndarray, t: int) -> np.ndarray:
    """Classical diffusion distance D_t(i,j)=sqrt(sum_a (P^t[i,a]-P^t[j,a])^2/pi[a])
    from the adaptive kNN kernel; pi is the degree-normalized stationary measure.
    Whitening by 1/sqrt(pi) turns the pi-weighted L2 into a plain cdist."""
    W = affinity_from_D(D0, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    Pt = np.linalg.matrix_power(P, t)
    pi = W.sum(1) / W.sum()
    Q = Pt / np.sqrt(np.maximum(pi, 1e-30))[None, :]
    return cdist(Q, Q)


def _heatgeo_radius(lam: np.ndarray, U: np.ndarray, D0: np.ndarray,
                    t: float, kmodes: int) -> float:
    """Median row-mass radius of the (truncated) heat kernel: for each i,
    sum_a mu_i(a) D0[i,a] with mu_i = rownorm(max(H_t[i,:],0))."""
    w = np.exp(-t * np.clip(lam[:kmodes], 0, None))
    Ht = (U[:, :kmodes] * w[None, :]) @ U[:, :kmodes].T
    M = np.clip(Ht, 0.0, None)
    rs = M.sum(1, keepdims=True)
    rs[rs <= 0] = 1.0
    return float(np.median(((M / rs) * D0).sum(1)))


def est_heatgeo(D0: np.ndarray) -> dict:
    """HeatGeo-style Varadhan distances for t (spread-matched) and 4t.

    t is chosen so the heat kernel's median row-mass radius ~ 0.3x the median
    ambient distance (the diffusion-spread rule); the search uses a truncated
    kernel for speed, the final distances the full kernel.
    """
    n = D0.shape[0]
    W = affinity_from_D(D0, k=KNN)
    W = 0.5 * (W + W.T)
    deg = W.sum(1)
    dm12 = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(n) - (dm12[:, None] * W * dm12[None, :])
    lam, U = np.linalg.eigh(0.5 * (L + L.T))
    lam = np.clip(lam, 0, None)

    off = D0[~np.eye(n, dtype=bool)]
    target = 0.3 * float(np.median(off))
    kmodes = min(n, 400)
    # coarse log grid then bisect between the bracketing grid points
    grid = np.geomspace(1e-3, 1e3, 16)
    rads = np.array([_heatgeo_radius(lam, U, D0, t, kmodes) for t in grid])
    if rads[-1] < target:
        t_star = grid[-1]
    elif rads[0] > target:
        t_star = grid[0]
    else:
        j = int(np.searchsorted(rads, target))
        lo, hi = grid[j - 1], grid[j]
        for _ in range(24):
            mid = np.sqrt(lo * hi)
            if _heatgeo_radius(lam, U, D0, mid, kmodes) < target:
                lo = mid
            else:
                hi = mid
        t_star = np.sqrt(lo * hi)

    out = {}
    for name, t in (("heatgeo_t", t_star), ("heatgeo_4t", 4.0 * t_star)):
        Ht = (U * np.exp(-t * lam)[None, :]) @ U.T
        Ht = 0.5 * (Ht + Ht.T)
        dhg = np.sqrt(np.maximum(-4.0 * t * np.log(np.maximum(Ht, 1e-300)), 0.0))
        np.fill_diagonal(dhg, 0.0)
        out[name] = 0.5 * (dhg + dhg.T)
    return out


def est_fb_potential(X: np.ndarray, seed: int) -> np.ndarray:
    """FB-potential distance: walks on the branch kNN graph -> seed-ensembled FB
    net -> per-seed potential distances -> averaged -> kNN(10) Dijkstra."""
    import pygsp
    from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
    from diffusion_curvature.trajectory_utils import subsample_trajectories
    G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)
    traj = subsample_trajectories(G, n_trajectories=100, length=50, rng=seed)
    coords = X[traj].astype(np.float32)
    ens = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=(7, 8, 9, 10), z_dim=64,
                            gamma=0.98, n_epochs=300, device=DEVICE)
    ens.fit(coords)
    K = np.maximum(ens.raw_kernels(X.astype(np.float32)), 0.0)   # (S, n, n)
    D_avg = np.mean([potential_distances(K[s]) for s in range(K.shape[0])], axis=0)
    return _graph(D_avg)


def compute_Ds(X: np.ndarray, d: int, seed: int, ests: list) -> dict:
    """All requested estimator distance matrices for one cloud, sharing cdist."""
    D0 = cdist(X, X)
    out = {}
    if "plain" in ests:
        out["plain"] = _graph(D0)
    if "pca_denoise" in ests:
        out["pca_denoise"] = est_pca_denoise(X, d)
    for t in (4, 8):
        name = f"diffdist_t{t}"
        if name in ests:
            out[name] = est_diffdist(D0, t)
    if "heatgeo_t" in ests or "heatgeo_4t" in ests:
        out.update(est_heatgeo(D0))
    if "fb_potential" in ests:
        out["fb_potential"] = est_fb_potential(X, seed)
    return {k: v for k, v in out.items() if k in ests}


# ---------------------------------------------------------------------------
# Flat-plane references (per branch x estimator), resumable
# ---------------------------------------------------------------------------

def _done_keys(path: Path, cols: list) -> set:
    if not (path.exists() and path.stat().st_size > 0):
        return set()
    df = pd.read_csv(path, usecols=cols)
    return set(map(tuple, df[cols].values))


def run_refs(ests: list, reps: int, fb_reps: int) -> None:
    key = ["branch", "estimator", "true_dim", "rep"]
    done = _done_keys(REFS_RAW_CSV, key) if REFS_RAW_CSV.exists() else set()
    header = not (REFS_RAW_CSV.exists() and REFS_RAW_CSV.stat().st_size > 0)
    t0 = time.time()
    for branch in BRANCHES:
        for d in DIMS:
            for est in ests:
                n_rep = fb_reps if est == "fb_potential" else reps
                for rep in range(n_rep):
                    if (branch, est, d, rep) in done:
                        continue
                    seed = 500 + 13 * d + rep
                    Xc = da.flat_plane(N, d, seed=seed)[0]
                    Xc = da.corrupt(Xc, branch, seed=seed)
                    D = compute_Ds(Xc, d, seed, [est])[est]
                    # anchor at the centroid-nearest point + 2 random interior pts
                    rng = np.random.default_rng(seed)
                    a0 = int(np.argmin(np.linalg.norm(Xc - Xc.mean(0), axis=1)))
                    anchors = np.array([a0, *rng.choice(N, 2, replace=False)])
                    fr = cs.extract_raw(D, anchors, d, device=DEVICE)
                    fr["branch"] = branch
                    fr["estimator"] = est
                    fr["rep"] = rep
                    fr["n_bucket"] = N
                    fr.to_csv(REFS_RAW_CSV, mode="a", index=False, header=header)
                    header = False
                    print(f"[refs] {branch} {est} d{d} rep{rep} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[refs] done ({time.time()-t0:.0f}s)")


def _ref_tables() -> dict:
    """(branch, estimator) -> mean/std table over (n_bucket, true_dim)."""
    raw = pd.read_csv(REFS_RAW_CSV)
    tables = {}
    for (branch, est), g in raw.groupby(["branch", "estimator"]):
        tables[(branch, est)] = g.groupby(["n_bucket", "true_dim"])[
            list(cs.CHANNELS)].agg(["mean", "std"])
    return tables


# ---------------------------------------------------------------------------
# Dumbbell + oracle-A points (per branch x estimator), resumable
# ---------------------------------------------------------------------------

def run_points(ests: list) -> None:
    key = ["kind", "dim", "seed", "branch", "estimator"]
    done = _done_keys(POINTS_CSV, key) if POINTS_CSV.exists() else set()
    header = not (POINTS_CSV.exists() and POINTS_CSV.stat().st_size > 0)
    t0 = time.time()
    for d in DIMS:
        for seed in SEEDS:
            f, L = dumbbell_profile(beta=DUMB_BETA)
            wp = WarpedProduct(f, L, d=d)
            Xclean, ks = da.sample_embedded(wp, N, d, seed=seed)
            eidx = v3.pick_eval_points(ks, N_EVAL, np.random.default_rng(seed))
            # oracle-A ceiling: exact geodesics, plain suite (estimator-free)
            if ("dumbbell", d, seed, "A", "oracle") not in done:
                m = wp.sample(N, rng=seed)
                Da, ksa = np.asarray(m["D"], float), np.asarray(m["ks_field"], float)
                ea = v3.pick_eval_points(ksa, N_EVAL, np.random.default_rng(seed))
                fr = cs.extract_raw(Da, ea, d, device=DEVICE)
                _emit(fr, ksa[ea], "dumbbell", d, seed, "A", "oracle",
                      POINTS_CSV, header)
                header = False
                print(f"[pts] A d{d} s{seed} ({time.time()-t0:.0f}s)", flush=True)
            for branch in BRANCHES:
                todo = [e for e in ests
                        if ("dumbbell", d, seed, branch, e) not in done]
                if not todo:
                    continue
                Xc = da.corrupt(Xclean, branch, seed=seed)
                Ds = compute_Ds(Xc, d, seed, todo)
                for est, D in Ds.items():
                    fr = cs.extract_raw(D, eidx, d, device=DEVICE)
                    _emit(fr, ks[eidx], "dumbbell", d, seed, branch, est,
                          POINTS_CSV, header)
                    header = False
                    print(f"[pts] {branch} {est} d{d} s{seed} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[pts] done ({time.time()-t0:.0f}s)")


def _emit(fr: pd.DataFrame, ks: np.ndarray, kind: str, d: int, seed: int,
          branch: str, est: str, path: Path, header: bool) -> None:
    fr = fr.copy()
    fr["ks"] = ks
    fr["kind"] = kind
    fr["dim"] = d
    fr["seed"] = seed
    fr["branch"] = branch
    fr["estimator"] = est
    fr.to_csv(path, mode="a", index=False, header=header)


# ---------------------------------------------------------------------------
# Summarize: z-score, integrator, tables
# ---------------------------------------------------------------------------

def _flatz(fr: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    df = fr.copy()
    for i, row in df.iterrows():
        k = (cs._bucket(int(row["n_points"])), int(row["true_dim"]))
        for c in cs.CHANNELS:
            try:
                mu = ref.loc[k, (c, "mean")]
                sd = ref.loc[k, (c, "std")]
            except KeyError:
                mu, sd = 0.0, 1.0
            df.at[i, c] = (row[c] - mu) / (abs(sd) + 1e-9)
    return df


def summarize() -> None:
    pts = pd.read_csv(POINTS_CSV)
    reftabs = _ref_tables()
    ref_tor = cs.load_flatref(embedded=False)
    model = joblib.load(PROC / "suite_model.joblib")

    rows = []
    for (kind, d, seed, branch, est), g in pts.groupby(
            ["kind", "dim", "seed", "branch", "estimator"]):
        ref = ref_tor if branch == "A" else reftabs.get((branch, est))
        if ref is None:
            continue
        frz = _flatz(g, ref)
        pr = cs.predict(frz, model)
        frz = frz.assign(integ_sign=pr["integ_sign"], integ_mag=pr["integ_mag"])
        rows.append(frz.assign(kind=kind, dim=d, seed=seed, branch=branch,
                               estimator=est))
    allz = pd.concat(rows, ignore_index=True)
    allz.to_csv(PROC / "distbk_p2_scored.csv", index=False)

    # --- per (d, branch, estimator): field r + balanced sign ---
    srows = []
    dumb = allz[allz.kind == "dumbbell"]
    for (d, branch, est), g in dumb.groupby(["dim", "branch", "estimator"]):
        rec = dict(dim=int(d), branch=branch, estimator=est, n=len(g))
        rec["r_integ"] = v3.pearson(g.integ_mag, g.ks)
        rec["sign_integ"] = v3.balanced_sign(g.integ_sign.values, g.ks.values)
        for c in REPORT:
            rec[f"r_{c}"] = v3.pearson(g[c], g.ks)
            rec[f"sign_{c}"] = v3.balanced_sign(g[c].values, g.ks.values)
        srows.append(rec)
    sdf = pd.DataFrame(srows).sort_values(["dim", "branch", "estimator"])
    sdf.to_csv(PROC / "distbk_p2_summary.csv", index=False)

    # --- flat-null hallucination: median |raw channel| on flat per estimator ---
    raw = pd.read_csv(REFS_RAW_CSV)
    fn = raw.groupby(["branch", "estimator", "true_dim"])[REPORT].agg(
        lambda s: float(np.median(np.abs(s))))
    fn.to_csv(PROC / "distbk_p2_flatnull.csv")

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print("\n=== field r + balanced sign per (d, branch, estimator) ===")
    print(sdf.round(3).to_string(index=False))
    print("\n=== flat-null hallucination: median |raw channel| on flat plane ===")
    print(fn.round(3).to_string())
    print("\n=== oracle-A ceiling (estimator-free, torus-referenced) ===")
    a = sdf[sdf.branch == "A"]
    print(a.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refs")
    r.add_argument("--estimators", default=",".join(ESTIMATORS))
    r.add_argument("--reps", type=int, default=4)
    r.add_argument("--fb-reps", type=int, default=2)
    p = sub.add_parser("run")
    p.add_argument("--estimators", default=",".join(ESTIMATORS))
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "refs":
        run_refs(args.estimators.split(","), args.reps, args.fb_reps)
    elif args.cmd == "run":
        run_points(args.estimators.split(","))
    elif args.cmd == "summarize":
        summarize()


if __name__ == "__main__":
    main()
