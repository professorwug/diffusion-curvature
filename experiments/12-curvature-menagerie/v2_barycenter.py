"""V2 barycenter-defect signed-curvature channel — cheap first-moment rider.

Using the V3 diffusion measure (cak n/4 kernel, t=2), the geodesic medoid of
mu_i is bar_i = argmin_z sum_a mu_i(a) d(z,a) (restricted to sample points).
The barycenter defect for a pair is

    b(i,j) = d(bar_i, bar_j) - d(i,j) .

In flat space the medoid of a symmetric diffusion cloud is the point itself, so
b = 0. Positive curvature pulls barycenters together (b < 0), so the per-point
score = mean of -b over the V3 pair set (16 pairs at 2x spread) is higher for
more positive curvature. Sign is mechanism-predicted and VERIFIED (not fit) on
the torus/sphere/hyperbolic triad.

Being a first-moment (mean-position) statistic, it is expected to be fragile at
boundaries; the `boundary` subcommand quantifies that on a bounded flat plane.

Usage:
  python v2_barycenter.py triad    [--quick]
  python v2_barycenter.py ladder
  python v2_barycenter.py boundary
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))

import v3_defect as v3  # noqa: E402
import v4_mds_strain as v4  # noqa: E402  (reuse cached build_manifold)

PROC = HERE / "processed_data"
PROC.mkdir(exist_ok=True)

CFG = dict(kernel="cak_n4", t=2, band=2.0)  # V3 measure config


def barycenter_scores(D: np.ndarray, eval_idx: np.ndarray,
                      n_pairs: int = 16) -> np.ndarray:
    """Per-eval-point score = mean of -(d(bar_i,bar_j) - d(i,j)) over the
    n_pairs points nearest radius band*spread."""
    P = v3.build_P(D, CFG["kernel"])
    M = v3.diffusion_powers(P, [CFG["t"]])[CFG["t"]]
    spread = v3.measure_spread(M, D)
    W = M @ D                       # W[i,z] = E_{a~mu_i} d(a,z)
    bar = np.argmin(W, axis=1)      # geodesic medoid of each measure
    target = CFG["band"] * spread
    scores = np.empty(len(eval_idx))
    for q, i in enumerate(eval_idx):
        diff = np.abs(D[i] - target); diff[i] = np.inf
        j = np.argpartition(diff, n_pairs)[:n_pairs]
        b = D[bar[i], bar[j]] - D[i, j]
        scores[q] = float(np.mean(-b))
    return scores


# ---------------------------------------------------------------------------
# Triad — verify the mechanism-predicted sign
# ---------------------------------------------------------------------------

def run_triad(quick: bool = False) -> None:
    n = 300 if quick else 1200
    n_eval = 12 if quick else 40
    seeds = [0] if quick else [0, 1]
    sign_manifolds = [("hyperbolic", 2), ("hyperbolic", 3),
                      ("torus", 2), ("torus", 3),
                      ("sphere", 2), ("sphere", 3)]
    field_manifolds = [("dumbbell", 3), ("necklace", 3)]
    t0 = time.time()
    rows = []
    for seed in seeds:
        for kind, d in sign_manifolds + field_manifolds:
            man = v4.build_manifold(kind, d, n, seed)
            D, ks = man["D"], man["ks"]
            eidx = v3.pick_eval_points(ks, n_eval, np.random.default_rng(1000 + seed))
            sc = barycenter_scores(D, eidx)
            for q, i in enumerate(eidx):
                rows.append(dict(kind=kind, dim=d, seed=seed,
                                 score=float(sc[q]), ks=float(ks[i])))
            print(f"[triad] seed{seed} {kind}{d} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "v2_bary_triad_points.csv", index=False)

    sph = df[df.kind == "sphere"].score.mean()
    hyp = df[df.kind == "hyperbolic"].score.mean()
    tor = df[df.kind == "torus"].score
    dumb = df[df.kind == "dumbbell"]; neck = df[df.kind == "necklace"]
    print(f"\n[triad] sphere(+K) mean={sph:+.4f}  torus(0) mean={tor.mean():+.4f}"
          f"  hyperbolic(-K) mean={hyp:+.4f}")
    print(f"[triad] mechanism sign holds (sph>hyp): {sph > hyp}")
    print(f"[triad] flat null: |score| med={tor.abs().median():.4f} "
          f"iqr={tor.quantile(.75)-tor.quantile(.25):.4f}")
    print(f"[triad] field r  dumbbell={v3.pearson(dumb.score, dumb.ks):.3f} "
          f"necklace={v3.pearson(neck.score, neck.ks):.3f}")


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

LADDER_DIMS = [3, 4, 5, 6]
LADDER_NOISE = [0.0, 0.15]


def run_ladder() -> None:
    n, n_eval = 1400, 40
    seeds = [0, 1]
    t0 = time.time()
    rows = []
    for kind in ("dumbbell", "necklace", "torus"):
        for d in LADDER_DIMS:
            for seed in seeds:
                man = v4.build_manifold(kind, d, n, seed)
                D0, ks = man["D"], man["ks"]
                eidx = v3.pick_eval_points(ks, n_eval, np.random.default_rng(2000 + seed))
                for noise in LADDER_NOISE:
                    D = v3._apply_noise(D0, noise, np.random.default_rng(3000 + seed))
                    sc = barycenter_scores(D, eidx)
                    for q, i in enumerate(eidx):
                        rows.append(dict(kind=kind, dim=d, seed=seed, noise=noise,
                                         p=int(i), score=float(sc[q]),
                                         ks=float(ks[i])))
            print(f"[ladder] {kind}{d} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "v2_bary_ladder_points.csv", index=False)

    v3ref = v4._v3_reference_scores(n, n_eval, seeds)
    summ = []
    for (kind, d, noise), g in df.groupby(["kind", "dim", "noise"]):
        rec = dict(kind=kind, dim=d, noise=noise, field_r=v3.pearson(g.score, g.ks))
        if kind in ("dumbbell", "necklace"):
            rec["balsign"] = v3.balanced_sign(g.score.values, g.ks.values)
            key = (kind, d, noise)
            if key in v3ref:
                merged = g.merge(v3ref[key], on=["seed", "p"])
                rec["corr_v3"] = v3.pearson(merged.score, merged.score_v3)
        else:
            rec["flatnull_med"] = float(g.score.abs().median())
            rec["flatnull_iqr"] = float(g.score.quantile(.75) - g.score.quantile(.25))
        summ.append(rec)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(PROC / "v2_bary_ladder_summary.csv", index=False)
    print(f"[ladder] done in {time.time()-t0:.0f}s")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    print(sdf.sort_values(["kind", "dim", "noise"]).to_string(index=False))


# ---------------------------------------------------------------------------
# Boundary check — bounded flat plane (first-moment fragility)
# ---------------------------------------------------------------------------

def run_boundary() -> None:
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(0)
    print("[boundary] bounded flat plane (should be a quiet null if robust):")
    for n in (400, 1200):
        X = rng.random((n, 2)) * 10.0
        D = cdist(X, X)
        s = np.median(D[~np.eye(n, dtype=bool)]); D /= s
        eidx = rng.choice(n, 40, replace=False)
        sc = barycenter_scores(D, eidx)
        # split by distance-to-centroid to expose boundary drift
        c = X.mean(0); rad = np.linalg.norm(X[eidx] - c, axis=1)
        edge = rad > np.median(rad)
        print(f"  n={n}: |score| med={np.median(np.abs(sc)):.4f} "
              f"mean={sc.mean():+.4f} | interior mean={sc[~edge].mean():+.4f} "
              f"edge mean={sc[edge].mean():+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triad"); t.add_argument("--quick", action="store_true")
    sub.add_parser("ladder")
    sub.add_parser("boundary")
    args = ap.parse_args()
    if args.cmd == "triad":
        run_triad(quick=args.quick)
    elif args.cmd == "ladder":
        run_ladder()
    elif args.cmd == "boundary":
        run_boundary()


if __name__ == "__main__":
    main()
