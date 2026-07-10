"""V4 local Gram/MDS-strain signed-curvature channel — staged benchmark.

For each evaluation point p we take its geodesic-ball patch of m nearest points,
double-center the patch's squared-distance submatrix (classical MDS Gram matrix
B = -1/2 J D2 J), and eigendecompose. In flat space B is PSD with rank exactly
the intrinsic dim d, so everything below the top-d eigenvalues vanishes.
Curvature bends geodesic distances away from Euclidean-embeddability and creates
a SIGNED spectral defect below rank d. Readouts (all cheap post-eigendecomp):

    (a) resid  : sum of eigenvalues ranked below top-d (signed; picks up the
                 negative tail that curvature induces)
    (b) resid_n: resid / sum(top-d eigenvalues)   (scale-free)
    (c) lam_d1 : the (d+1)-th eigenvalue alone    (signed)

The mapping readout -> curvature sign is DERIVED on the torus/sphere/hyperbolic
triad (`triad` subcommand), pre-registered (sign_factor so higher = more +K),
and frozen before the ladder (`ladder` subcommand).

Usage:
  python v4_mds_strain.py triad   [--quick]
  python v4_mds_strain.py ladder
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))

import v3_defect as v3  # noqa: E402
from diffusion_curvature.menagerie import (  # noqa: E402
    WarpedProduct, dumbbell_profile, necklace_profile, torus_flat, sphere_sf,
    hyperbolic,
)

PROC = HERE / "processed_data"
PROC.mkdir(exist_ok=True)
CACHE = PROC / "manifold_cache"
CACHE.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Cached manifold construction (extends v3 with hyperbolic; same normalization)
# ---------------------------------------------------------------------------

def build_manifold(kind: str, d: int, n: int, seed: int) -> dict:
    key = CACHE / f"{kind}_d{d}_n{n}_s{seed}.joblib"
    if key.exists():
        return joblib.load(key)
    rng = np.random.default_rng(seed)
    if kind == "hyperbolic":
        m = hyperbolic(n, d, kappa=1.0, R=2.0, rng=rng)
        D, ks = v3._normalize(np.asarray(m["D"], float),
                              np.asarray(m["ks_field"], float))
        out = dict(D=D, ks=ks, kind=kind, dim=d)
    else:
        out = v3.build_manifold(kind, d, n, seed)
    joblib.dump(out, key)
    return out


# ---------------------------------------------------------------------------
# MDS-strain readouts
# ---------------------------------------------------------------------------

def gram_eigs(D2sub: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the double-centered Gram matrix."""
    m = D2sub.shape[0]
    J = np.eye(m) - 1.0 / m
    B = -0.5 * (J @ D2sub @ J)
    B = 0.5 * (B + B.T)
    return np.linalg.eigvalsh(B)[::-1]


def estimate_rank(eigs: np.ndarray, dmax: int) -> int:
    """Top-eigengap heuristic: largest consecutive ratio among leading eigs."""
    top = np.maximum(eigs[:2 * dmax + 3], 1e-12)
    if len(top) < 2:
        return 1
    gaps = top[:-1] / top[1:]
    return int(np.argmax(gaps[:dmax + 2]) + 1)


def readouts(eigs: np.ndarray, d: int) -> dict[str, float]:
    d = max(1, min(d, len(eigs) - 1))
    top = eigs[:d]
    resid = eigs[d:]
    return dict(
        resid=float(resid.sum()),
        resid_n=float(resid.sum() / max(top.sum(), 1e-12)),
        lam_d1=float(eigs[d]),
    )


def point_readouts(D2: np.ndarray, order: np.ndarray, p: int, m: int,
                   pool: str, true_dim: int, rng) -> dict[tuple, float]:
    """Readouts at point p for each (readout_name, dest) key; jitter pooling
    averages readout values over 4 patches around p."""
    if pool == "single":
        patches = [order[p, :m]]
    else:
        big = order[p, :min(2 * m, D2.shape[0])]
        patches = [order[p, :m]]
        for _ in range(3):
            extra = rng.choice(big[1:], size=m - 1, replace=False)
            patches.append(np.concatenate([[p], extra]))
    acc: dict[tuple, list] = {}
    for patch in patches:
        eigs = gram_eigs(D2[np.ix_(patch, patch)])
        for dest, dd in (("true", true_dim), ("est", estimate_rank(eigs, true_dim))):
            for name, val in readouts(eigs, dd).items():
                acc.setdefault((name, dest), []).append(val)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Shared per-cell scoring
# ---------------------------------------------------------------------------

MS = [30, 60, 120]
POOLS = ["single", "jitter"]
DISTS = ["exact", "graph"]


def score_cell(D: np.ndarray, ks: np.ndarray, eval_idx: np.ndarray,
               m: int, pool: str, true_dim: int, seed: int) -> list[dict]:
    D2 = D ** 2
    order = np.argsort(D, axis=1)
    rng = np.random.default_rng(9000 + seed)
    rows = []
    for p in eval_idx:
        ro = point_readouts(D2, order, int(p), m, pool, true_dim, rng)
        for (name, dest), val in ro.items():
            rows.append(dict(readout=name, dest=dest, m=m, pool=pool,
                             p=int(p), value=val, ks=float(ks[p])))
    return rows


# ---------------------------------------------------------------------------
# Triad — derive & freeze the sign convention
# ---------------------------------------------------------------------------

def run_triad(quick: bool = False) -> None:
    n = 300 if quick else 1200
    n_eval = 12 if quick else 40
    seeds = [0] if quick else [0, 1]
    ms = [60] if quick else MS
    pools = ["single"] if quick else POOLS
    dists = ["exact"] if quick else DISTS
    # sign manifolds (constant ks): hyperbolic (-K), torus (0), sphere (+K);
    # plus the signed warped fields for the field-r check.
    sign_manifolds = [("hyperbolic", 2), ("hyperbolic", 3),
                      ("torus", 2), ("torus", 3),
                      ("sphere", 2), ("sphere", 3)]
    field_manifolds = [("dumbbell", 3), ("necklace", 3)]
    all_manifolds = sign_manifolds + field_manifolds

    t0 = time.time()
    rows = []
    for seed in seeds:
        for kind, d in all_manifolds:
            man = build_manifold(kind, d, n, seed)
            D0, ks = man["D"], man["ks"]
            eidx = v3.pick_eval_points(ks, n_eval, np.random.default_rng(1000 + seed))
            for dist in dists:
                D = D0 if dist == "exact" else v3._graph_geodesic(D0, k=10)
                for m in ms:
                    for pool in pools:
                        for r in score_cell(D, ks, eidx, m, pool, d, seed):
                            r.update(kind=kind, dim=d, seed=seed, dist=dist)
                            rows.append(r)
            print(f"[triad] seed{seed} {kind}{d} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "v4_mds_triad_points.csv", index=False)

    # derive sign per (readout, dest, m, pool, dist): sphere mean vs hyperbolic mean
    summ = []
    keys = ["readout", "dest", "m", "pool", "dist"]
    for key, g in df.groupby(keys):
        sph = g[g.kind == "sphere"].value.mean()
        hyp = g[g.kind == "hyperbolic"].value.mean()
        tor_all = g[g.kind == "torus"].value
        sign = 1.0 if (sph - hyp) >= 0 else -1.0
        # apply frozen sign; monotonicity target is hyp < tor < sph
        s_hyp, s_tor, s_sph = sign * hyp, sign * tor_all.mean(), sign * sph
        dumb = g[g.kind == "dumbbell"]
        neck = g[g.kind == "necklace"]
        rec = dict(zip(keys, key))
        rec.update(
            sign=sign,
            monotone=bool(s_hyp < s_tor < s_sph),
            triad_gap=float(s_sph - s_hyp),
            null_med=float((sign * tor_all).abs().median()),
            null_iqr=float((sign * tor_all).quantile(0.75)
                           - (sign * tor_all).quantile(0.25)),
            field_r_dumbbell=v3.pearson(sign * dumb.value, dumb.ks),
            field_r_necklace=v3.pearson(sign * neck.value, neck.ks),
        )
        summ.append(rec)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(PROC / "v4_mds_triad_summary.csv", index=False)
    print(f"[triad] {len(sdf)} cells in {time.time()-t0:.0f}s")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    good = sdf[sdf.monotone].copy()
    good["field_min"] = good[["field_r_dumbbell", "field_r_necklace"]].min(axis=1)
    print("\nMonotone cells, ranked by min field r (then triad gap):")
    print(good.sort_values(["field_min", "triad_gap"], ascending=False)
          .head(14).to_string(index=False))


# ---------------------------------------------------------------------------
# Ladder — frozen config
# ---------------------------------------------------------------------------

# Frozen after triad derivation (see run_triad output). Placeholder defaults
# overwritten by _load_frozen() from the triad summary.
FROZEN = dict(readout="resid_n", dest="true", m=60, pool="single", sign=1.0)


def _load_frozen() -> dict:
    path = PROC / "v4_mds_triad_summary.csv"
    if not path.exists():
        return FROZEN
    s = pd.read_csv(path)
    s = s[s.monotone & (s.dist == "exact")].copy()
    if not len(s):
        return FROZEN
    s["field_min"] = s[["field_r_dumbbell", "field_r_necklace"]].min(axis=1)
    best = s.sort_values(["field_min", "triad_gap"], ascending=False).iloc[0]
    return dict(readout=best.readout, dest=best.dest, m=int(best.m),
                pool=best.pool, sign=float(best.sign))


LADDER_DIMS = [3, 4, 5, 6]
LADDER_NOISE = [0.0, 0.15]

# Frozen after triad: readout=resid_n (scale-free), sign=-1 (score=-residual;
# derived sphere<hyperbolic), true dim. Patch size m swept on the ladder because
# it is the one hyperparameter expected to matter at high dim. Deterministic
# (single pool, exact distances -> analytically zero flat null).
LADDER_CONFIGS = [
    dict(readout="resid_n", dest="true", m=30, pool="single", sign=-1.0),
    dict(readout="resid_n", dest="true", m=60, pool="single", sign=-1.0),
    dict(readout="resid_n", dest="true", m=120, pool="single", sign=-1.0),
]


def run_ladder(frozen: dict | None = None) -> None:
    configs = [frozen] if frozen else LADDER_CONFIGS
    n, n_eval = 1400, 40
    seeds = [0, 1]
    manifolds = [("dumbbell",), ("necklace",), ("torus",)]
    t0 = time.time()
    rows = []
    for (kind,) in manifolds:
        for d in LADDER_DIMS:
            for seed in seeds:
                man = build_manifold(kind, d, n, seed)
                D0, ks = man["D"], man["ks"]
                eidx = v3.pick_eval_points(ks, n_eval, np.random.default_rng(2000 + seed))
                for noise in LADDER_NOISE:
                    D = v3._apply_noise(D0, noise, np.random.default_rng(3000 + seed))
                    for cfg in configs:
                        cname = f"{cfg['readout']}|{cfg['dest']}|m{cfg['m']}|{cfg['pool']}"
                        cell = score_cell(D, ks, eidx, cfg["m"], cfg["pool"], d, seed)
                        cell = [r for r in cell
                                if r["readout"] == cfg["readout"]
                                and r["dest"] == cfg["dest"]]
                        for r in cell:
                            rows.append(dict(config=cname, kind=kind, dim=d,
                                             seed=seed, noise=noise, p=r["p"],
                                             score=cfg["sign"] * r["value"],
                                             ks=r["ks"]))
            print(f"[ladder] {kind}{d} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "v4_mds_ladder_points.csv", index=False)

    # cross-correlate with V3 static score per cell (recompute V3 headline)
    v3scores = _v3_reference_scores(n, n_eval, seeds)

    summ = []
    for (config, kind, d, noise), g in df.groupby(["config", "kind", "dim", "noise"]):
        rec = dict(config=config, kind=kind, dim=d, noise=noise,
                   field_r=v3.pearson(g.score, g.ks))
        if kind in ("dumbbell", "necklace"):
            rec["balsign"] = v3.balanced_sign(g.score.values, g.ks.values)
            key = (kind, d, noise)
            if key in v3scores:
                # align by point p across seeds
                vv = v3scores[key]
                merged = g.merge(vv, on=["seed", "p"], suffixes=("", "_v3"))
                rec["corr_v3"] = v3.pearson(merged.score, merged.score_v3)
        else:
            rec["flatnull_med"] = float(g.score.abs().median())
            rec["flatnull_iqr"] = float(g.score.quantile(0.75) - g.score.quantile(0.25))
        summ.append(rec)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(PROC / "v4_mds_ladder_summary.csv", index=False)
    print(f"[ladder] done in {time.time()-t0:.0f}s")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    print(sdf.sort_values(["config", "kind", "dim", "noise"]).to_string(index=False))


def _v3_reference_scores(n: int, n_eval: int, seeds: list[int]) -> dict:
    """V3 headline static score (cak_n4/t2/b2.0/raw) per (kind,d,noise), keyed
    for point-aligned cross-correlation."""
    cfg = dict(kernel="cak_n4", t=2, band=2.0, norm="raw")
    out = {}
    for kind in ("dumbbell", "necklace"):
        for d in LADDER_DIMS:
            for noise in LADDER_NOISE:
                recs = []
                for seed in seeds:
                    man = build_manifold(kind, d, n, seed)
                    D0, ks = man["D"], man["ks"]
                    eidx = v3.pick_eval_points(ks, n_eval,
                                               np.random.default_rng(2000 + seed))
                    D = v3._apply_noise(D0, noise, np.random.default_rng(3000 + seed))
                    D2 = D ** 2
                    P = v3.build_P(D, cfg["kernel"])
                    M = v3.diffusion_powers(P, [cfg["t"]])[cfg["t"]]
                    delta = v3.defect_matrix(M, D2)
                    spread = v3.measure_spread(M, D)
                    sc = v3.point_scores(delta, D, eidx, cfg["band"] * spread,
                                         cfg["norm"], spread, 16)
                    for q, i in enumerate(eidx):
                        recs.append(dict(seed=seed, p=int(i), score_v3=sc[q]))
                out[(kind, d, noise)] = pd.DataFrame(recs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triad"); t.add_argument("--quick", action="store_true")
    sub.add_parser("ladder")
    args = ap.parse_args()
    if args.cmd == "triad":
        run_triad(quick=args.quick)
    elif args.cmd == "ladder":
        run_ladder()


if __name__ == "__main__":
    main()
