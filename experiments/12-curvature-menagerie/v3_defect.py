"""V3 second-moment *defect* signed-curvature estimator — staged benchmark.

For a point i with diffusion measure mu_i (row i of P^t) and a paired point j
at moderate radius, the second-moment defect is

    delta(i,j) = E_{x~mu_i,y~mu_j} dg2(x,y)
                 - 1/2 E_{x,x'~mu_i} dg2(x,x')
                 - 1/2 E_{y,y'~mu_j} dg2(y,y')
                 - dg(i,j)^2 ,

with dg2 the elementwise-squared geodesic distance matrix. All expectations
are EXACT matrix contractions. Writing A = M D2 M^T (M = P^t) gives

    delta(i,j) = A[i,j] - 1/2 A[i,i] - 1/2 A[j,j] - D[i,j]^2 .

In flat Euclidean space delta = 0 exactly (second-moment identity). Signed
convention: delta < 0 <=> positive curvature (measures contract). The per-point
score is the mean of -delta over n_pairs pairs j in a target radius band, so
higher score = more positive curvature (matches the repo ks convention).

Stages are subcommands: `stage-a`, `stage-b`, `stage-c`.

Usage:
  python v3_defect.py stage-a   [--quick]
  python v3_defect.py stage-b
  python v3_defect.py stage-c
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))

from diffusion_curvature.menagerie import (  # noqa: E402
    WarpedProduct, dumbbell_profile, necklace_profile, torus_flat, sphere_sf,
)
from benchmark_kmetric_colosseum import affinity_from_D  # noqa: E402
from graph_ablation_ruler import cak_W  # noqa: E402

PROC = HERE / "processed_data"
PROC.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Manifold construction (exact geodesic distances; certified ks fields)
# ---------------------------------------------------------------------------

def _normalize(D: np.ndarray, ks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale so median off-diagonal distance = 1; ks carries units 1/length^2."""
    off = D[~np.eye(D.shape[0], dtype=bool)]
    s = float(np.median(off))
    return D / s, ks * s**2


def build_manifold(kind: str, d: int, n: int, seed: int) -> dict:
    """Return dict(D, ks, kind, dim) with normalized exact geodesics."""
    rng = np.random.default_rng(seed)
    if kind == "torus":
        m = torus_flat(n, d, rng=rng)
    elif kind == "sphere":
        m = sphere_sf(n, d, r=1.0, rng=rng)
    elif kind == "dumbbell":
        f, L = dumbbell_profile(beta=0.8)
        m = WarpedProduct(f, L, d).sample(n, rng=rng)
    elif kind == "necklace":
        f, L = necklace_profile(b=0.7)
        m = WarpedProduct(f, L, d, periodic=True).sample(n, rng=rng)
    else:
        raise ValueError(kind)
    D, ks = _normalize(np.asarray(m["D"], float), np.asarray(m["ks_field"], float))
    return dict(D=D, ks=ks, kind=kind, dim=d)


# ---------------------------------------------------------------------------
# Kernels and diffusion powers
# ---------------------------------------------------------------------------

def build_P(D: np.ndarray, kernel: str) -> np.ndarray:
    """Row-stochastic diffusion operator from a distance matrix."""
    if kernel == "adaptive_k10":
        W = affinity_from_D(D, k=10)
    elif kernel == "adaptive_k25":
        W = affinity_from_D(D, k=25)
    elif kernel == "cak_n4":
        W, _ = cak_W(D, target=max(4, D.shape[0] // 4))
    else:
        raise ValueError(kernel)
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30)


def diffusion_powers(P: np.ndarray, t_list: list[int]) -> dict[int, np.ndarray]:
    """Return {t: P^t} for t a set of powers of two, via repeated squaring."""
    powers: dict[int, np.ndarray] = {}
    Mp, tp = P, 1
    while tp <= max(t_list):
        tp2 = tp * 2
        Mp = Mp @ Mp
        tp = tp2
        if tp in t_list:
            powers[tp] = Mp
    return powers


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def defect_matrix(M: np.ndarray, D2: np.ndarray) -> np.ndarray:
    """delta[i,j] = A[i,j] - 0.5 A[i,i] - 0.5 A[j,j] - D2[i,j], A = M D2 M^T."""
    A = M @ D2 @ M.T
    a = np.diag(A)
    return A - 0.5 * a[:, None] - 0.5 * a[None, :] - D2


def measure_spread(M: np.ndarray, D: np.ndarray) -> float:
    """spread = mean_i sum_a mu_i(a) D[i,a] (mean one-step measure radius)."""
    return float(np.mean((M * D).sum(axis=1)))


def point_scores(delta: np.ndarray, D: np.ndarray, eval_idx: np.ndarray,
                 target: float, norm: str, spread: float,
                 n_pairs: int) -> np.ndarray:
    """Per-eval-point score: mean of -delta over the n_pairs points nearest the
    target radius, with the chosen normalization."""
    n = D.shape[0]
    scores = np.empty(len(eval_idx))
    for q, i in enumerate(eval_idx):
        diff = np.abs(D[i] - target)
        diff[i] = np.inf
        j = np.argpartition(diff, n_pairs)[:n_pairs]
        val = -delta[i, j]
        if norm == "spread2":
            val = val / spread**2
        elif norm == "dij2":
            val = val / np.maximum(D[i, j]**2, 1e-12)
        scores[q] = float(np.mean(val))
    return scores


def pick_eval_points(ks: np.ndarray, n_eval: int, rng) -> np.ndarray:
    """Stratify eval points across the ks range (random if ks is constant)."""
    n = len(ks)
    n_eval = min(n_eval, n)
    if np.ptp(ks) < 1e-9:
        return rng.choice(n, n_eval, replace=False)
    order = np.argsort(ks)
    sel = np.linspace(0, n - 1, n_eval).round().astype(int)
    return order[sel]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def balanced_sign(scores: np.ndarray, ks: np.ndarray) -> float:
    """Balanced accuracy of sign(score) vs sign(ks) at the native zero."""
    scores, ks = np.asarray(scores, float), np.asarray(ks, float)
    pos, neg = ks > 0, ks < 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    tpr = float(np.mean(scores[pos] > 0))
    tnr = float(np.mean(scores[neg] < 0))
    return 0.5 * (tpr + tnr)


# ---------------------------------------------------------------------------
# Stage A — triads grid
# ---------------------------------------------------------------------------

STAGE_A_MANIFOLDS = [
    ("torus", 2), ("torus", 3),
    ("sphere", 2), ("sphere", 3),
    ("dumbbell", 3), ("necklace", 3),
]
KERNELS = ["adaptive_k10", "adaptive_k25", "cak_n4"]
T_LIST = [2, 4, 8, 16]
BANDS = [0.5, 1.0, 2.0]
NORMS = ["raw", "spread2", "dij2"]


def run_stage_a(quick: bool = False) -> None:
    n = 300 if quick else 1200
    n_eval = 12 if quick else 40
    n_pairs = 16
    seeds = [0] if quick else [0, 1]
    kernels = ["adaptive_k10"] if quick else KERNELS
    t_list = [4, 8] if quick else T_LIST

    t0 = time.time()
    # collect per-(seed,manifold,kernel,t,band,norm) scores keyed for aggregation
    rows = []  # long-form per eval-point, aggregated afterwards
    for seed in seeds:
        for kind, d in STAGE_A_MANIFOLDS:
            man = build_manifold(kind, d, n, seed)
            D, ks = man["D"], man["ks"]
            D2 = D**2
            rng = np.random.default_rng(1000 + seed)
            eval_idx = pick_eval_points(ks, n_eval, rng)
            for kernel in kernels:
                P = build_P(D, kernel)
                powers = diffusion_powers(P, t_list)
                for t in t_list:
                    M = powers[t]
                    delta = defect_matrix(M, D2)
                    spread = measure_spread(M, D)
                    for band in BANDS:
                        target = band * spread
                        for norm in NORMS:
                            sc = point_scores(delta, D, eval_idx, target,
                                              norm, spread, n_pairs)
                            for q, i in enumerate(eval_idx):
                                rows.append(dict(
                                    seed=seed, kind=kind, dim=d, kernel=kernel,
                                    t=t, band=band, norm=norm,
                                    score=sc[q], ks=float(ks[i])))
            print(f"[stage-a] seed{seed} {kind}{d} done "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "v3_defect_stageA_points.csv", index=False)

    # aggregate into config grid
    grid = []
    for (kernel, t, band, norm), g in df.groupby(["kernel", "t", "band", "norm"]):
        dumb = g[g.kind == "dumbbell"]
        neck = g[g.kind == "necklace"]
        sph = g[g.kind == "sphere"]
        tor = g[g.kind == "torus"]
        pooled = pd.concat([dumb, neck])
        grid.append(dict(
            kernel=kernel, t=t, band=band, norm=norm,
            field_r_dumbbell=pearson(dumb.score, dumb.ks),
            field_r_necklace=pearson(neck.score, neck.ks),
            sep_sphere_torus=float(sph.score.mean() - tor.score.mean()),
            flatnull_med=float(tor.score.abs().median()),
            flatnull_iqr=float(tor.score.quantile(0.75)
                               - tor.score.quantile(0.25)),
            balsign=balanced_sign(pooled.score.values, pooled.ks.values),
        ))
    gdf = pd.DataFrame(grid)
    gdf.to_csv(PROC / "v3_defect_stageA_grid.csv", index=False)
    print(f"[stage-a] {len(gdf)} configs in {time.time()-t0:.0f}s")
    print("\nTop configs by balanced sign (field r >= 0.4 both families):")
    ok = gdf[(gdf.field_r_dumbbell >= 0.4) & (gdf.field_r_necklace >= 0.4)]
    show = (ok if len(ok) else gdf).sort_values("balsign", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(show.head(12).to_string(index=False))


# ---------------------------------------------------------------------------
# Stage B — dimension x noise sweep on best configs
# ---------------------------------------------------------------------------

STAGE_B_DIMS = [3, 4, 5, 6]
STAGE_B_NOISE = [0.0, 0.05, 0.15]


def _apply_noise(D: np.ndarray, noise: float, rng) -> np.ndarray:
    if noise <= 0:
        return D
    E = rng.normal(size=D.shape)
    E = 0.5 * (E + E.T)
    Dn = D * (1.0 + noise * E)
    Dn = np.maximum(Dn, 0.0)
    np.fill_diagonal(Dn, 0.0)
    return 0.5 * (Dn + Dn.T)


# Stage-A verdict: cak_n4 is the only kernel with a quiet null; raw norm keeps
# it quietest at equal balanced-sign; t=2/band=2.0 is the sweet spot. The t4/b1
# row probes robustness; the dij2 row shows the normalization trade-off.
STAGE_B_CONFIGS = [
    dict(kernel="cak_n4", t=2, band=2.0, norm="raw"),
    dict(kernel="cak_n4", t=4, band=1.0, norm="raw"),
    dict(kernel="cak_n4", t=2, band=2.0, norm="dij2"),
]


def run_stage_b(configs: list[dict] | None = None) -> None:
    if configs is None:
        configs = STAGE_B_CONFIGS
    n, n_eval, n_pairs = 1400, 40, 16
    seeds = [0, 1]
    manifolds = [("dumbbell",), ("necklace",), ("torus",)]

    t0 = time.time()
    rows = []
    for cfg in configs:
        cname = f"{cfg['kernel']}|t{cfg['t']}|b{cfg['band']}|{cfg['norm']}"
        for (kind,) in manifolds:
            for d in STAGE_B_DIMS:
                for seed in seeds:
                    man = build_manifold(kind, d, n, seed)
                    D0, ks = man["D"], man["ks"]
                    rng = np.random.default_rng(2000 + seed)
                    eval_idx = pick_eval_points(ks, n_eval, rng)
                    for noise in STAGE_B_NOISE:
                        nrng = np.random.default_rng(3000 + seed)
                        D = _apply_noise(D0, noise, nrng)
                        D2 = D**2
                        P = build_P(D, cfg["kernel"])
                        powers = diffusion_powers(P, [cfg["t"]])
                        M = powers[cfg["t"]]
                        delta = defect_matrix(M, D2)
                        spread = measure_spread(M, D)
                        sc = point_scores(delta, D, eval_idx,
                                          cfg["band"] * spread, cfg["norm"],
                                          spread, n_pairs)
                        for q, i in enumerate(eval_idx):
                            rows.append(dict(
                                config=cname, kind=kind, dim=d, noise=noise,
                                seed=seed, score=sc[q], ks=float(ks[i])))
                print(f"[stage-b] {cname} {kind}{d} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "v3_defect_stageB_points.csv", index=False)

    summ = []
    for (config, kind, d, noise), g in df.groupby(
            ["config", "kind", "dim", "noise"]):
        rec = dict(config=config, kind=kind, dim=d, noise=noise,
                   field_r=pearson(g.score, g.ks))
        if kind in ("dumbbell", "necklace"):
            rec["balsign"] = balanced_sign(g.score.values, g.ks.values)
        else:  # torus null
            rec["flatnull_med"] = float(g.score.abs().median())
            rec["flatnull_iqr"] = float(g.score.quantile(0.75)
                                        - g.score.quantile(0.25))
        summ.append(rec)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(PROC / "v3_defect_stageB_summary.csv", index=False)
    print(f"[stage-b] done in {time.time()-t0:.0f}s")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(sdf.sort_values(["config", "kind", "dim", "noise"]).to_string(index=False))


def _load_best_configs() -> list[dict]:
    """Pick 2-3 Stage-A configs: best balanced sign among those with a quiet
    null and non-trivial field correlation."""
    path = PROC / "v3_defect_stageA_grid.csv"
    g = pd.read_csv(path)
    # require both families correlate and separation is positive (right sign)
    ok = g[(g.field_r_dumbbell >= 0.4) & (g.field_r_necklace >= 0.4)
           & (g.sep_sphere_torus > 0)]
    if len(ok) < 2:
        ok = g[(g.field_r_necklace >= 0.4) & (g.sep_sphere_torus > 0)]
    ok = ok.sort_values("balsign", ascending=False).head(3)
    return ok[["kernel", "t", "band", "norm"]].to_dict("records")


# ---------------------------------------------------------------------------
# Stage C — graph-geodesic bias check (Dijkstra over kNN of D)
# ---------------------------------------------------------------------------

def _graph_geodesic(D: np.ndarray, k: int = 10) -> np.ndarray:
    """Dijkstra shortest paths over a kNN graph built from D (approx geodesics)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    wts = D[rows, cols]
    G = sp.csr_matrix((wts, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    Dg = dijkstra(G, directed=False)
    # patch disconnected pairs with the exact distance (rare)
    bad = ~np.isfinite(Dg)
    Dg[bad] = D[bad]
    return 0.5 * (Dg + Dg.T)


def run_stage_c(configs: list[dict] | None = None) -> None:
    if configs is None:
        configs = [dict(kernel="cak_n4", t=2, band=2.0, norm="raw")]
    n, n_eval, n_pairs = 1400, 40, 16
    seeds = [0, 1]
    t0 = time.time()
    rows = []
    for cfg in configs:
        cname = f"{cfg['kernel']}|t{cfg['t']}|b{cfg['band']}|{cfg['norm']}"
        for kind in ("dumbbell", "necklace"):
            for d in (3, 4):
                for seed in seeds:
                    man = build_manifold(kind, d, n, seed)
                    D_exact, ks = man["D"], man["ks"]
                    rng = np.random.default_rng(4000 + seed)
                    eval_idx = pick_eval_points(ks, n_eval, rng)
                    for ground in ("exact", "graph"):
                        D = D_exact if ground == "exact" else \
                            _graph_geodesic(D_exact, k=10)
                        D2 = D**2
                        P = build_P(D, cfg["kernel"])
                        M = diffusion_powers(P, [cfg["t"]])[cfg["t"]]
                        delta = defect_matrix(M, D2)
                        spread = measure_spread(M, D)
                        sc = point_scores(delta, D, eval_idx,
                                          cfg["band"] * spread, cfg["norm"],
                                          spread, n_pairs)
                        for q, i in enumerate(eval_idx):
                            rows.append(dict(
                                config=cname, kind=kind, dim=d, ground=ground,
                                seed=seed, score=sc[q], ks=float(ks[i])))
            print(f"[stage-c] {cname} {kind} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "v3_defect_stageC_points.csv", index=False)
    summ = []
    for (config, kind, d, ground), g in df.groupby(
            ["config", "kind", "dim", "ground"]):
        summ.append(dict(config=config, kind=kind, dim=d, ground=ground,
                         field_r=pearson(g.score, g.ks),
                         balsign=balanced_sign(g.score.values, g.ks.values)))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(PROC / "v3_defect_stageC_summary.csv", index=False)
    print(f"[stage-c] done in {time.time()-t0:.0f}s")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(sdf.sort_values(["kind", "dim", "ground"]).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage-a")
    a.add_argument("--quick", action="store_true")
    sub.add_parser("stage-b")
    sub.add_parser("stage-c")
    args = ap.parse_args()
    if args.cmd == "stage-a":
        run_stage_a(quick=args.quick)
    elif args.cmd == "stage-b":
        run_stage_b()
    elif args.cmd == "stage-c":
        run_stage_c()


if __name__ == "__main__":
    main()
