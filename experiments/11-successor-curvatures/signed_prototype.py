"""Sanity prototype for the natively signed Wasserstein curvatures.

Pass criteria:
  1. sign: sphere > 0, saddle < 0, plane ~ 0 (both estimators)
  2. magnitude: kappa monotone in 1/r^2 across sphere radii
  3. stability across t

Usage: pixi run python signed_prototype.py [--quick]
Writes processed_data/signed_prototype.csv
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from diffusion_curvature.datasets import plane, rejection_sample_from_saddle, sphere
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

warnings.filterwarnings("ignore")


def make_datasets(n: int, d: int, seed: int) -> list[tuple[str, np.ndarray, float, list[int]]]:
    """Returns (name, X, ks_true, eval_idxs).

    eval_idxs: indices at which the true curvature applies (origin for
    saddle/plane; any point for spheres -- we use several for averaging).
    """
    rng = np.random.default_rng(seed)
    out = []

    Xp = plane(n, dim=d)
    Xp = np.hstack([Xp, np.zeros((n, 1))])
    out.append(("plane", Xp, 0.0, [0]))

    for r in (1.0, 1.5, 2.0):
        Xs, _ = sphere(n, d=d, seed=seed)
        Xs = Xs * r
        ks = d * (d - 1) / r**2
        idxs = rng.choice(n, size=4, replace=False).tolist()
        out.append((f"sphere_r{r}", Xs, ks, idxs))

    (Xsad, ksad) = rejection_sample_from_saddle(n, d)
    out.append(("saddle", np.asarray(Xsad), float(ksad), [0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--ts", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--out", default="processed_data/signed_prototype.csv")
    args = ap.parse_args()
    if args.quick:
        args.n, args.dims, args.ts = 1000, [2], [4, 8]

    rows = []
    for d in args.dims:
        datasets = make_datasets(args.n, d, seed=42)
        for name, X, ks, idxs in datasets:
            for t in args.ts:
                est = WassersteinSignedCurvature(t=t, knn=10, n_pairs=8, seed=0)
                t0 = time.time()
                est.fit(X=X.astype(np.float64), idx=idxs)
                el = time.time() - t0
                row = dict(
                    dim=d, dataset=name, t=t, ks_true=ks,
                    orc=float(np.nanmean(est.orc_)),
                    orc_std=float(np.nanstd(est.orc_)),
                    orc_phys=float(np.nanmean(est.orc_phys_)),
                    midpoint=float(np.nanmean(est.midpoint_)),
                    midpoint_std=float(np.nanstd(est.midpoint_)),
                    spread=float(np.nanmean(est.spread_)),
                    elapsed_s=round(el, 2),
                )
                rows.append(row)
                print(
                    f"d={d} {name:<11} t={t:<3} ks={ks:+7.2f}  "
                    f"orc={row['orc']:+.4f}±{row['orc_std']:.4f}  "
                    f"phys={row['orc_phys']:+7.3f}  "
                    f"mid={row['midpoint']:+.4f}±{row['midpoint_std']:.4f}  "
                    f"({el:.1f}s)", flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # quick verdicts
    for d in args.dims:
        sub = df[df.dim == d]
        for t in args.ts:
            st = sub[sub.t == t]
            sphere_pos = (st[st.dataset.str.startswith("sphere")].orc > 0).all()
            saddle_neg = (st[st.dataset == "saddle"].orc < 0).all()
            pl = st[st.dataset == "plane"].orc.abs().max()
            sp = st[st.dataset.str.startswith("sphere")].sort_values("ks_true")
            mono = sp.orc.is_monotonic_increasing
            print(f"d={d} t={t}: ORC sphere>0:{sphere_pos} saddle<0:{saddle_neg} "
                  f"|plane|={pl:.4f} radius-monotone:{mono}")


if __name__ == "__main__":
    main()
