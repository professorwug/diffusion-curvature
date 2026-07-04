"""Menagerie battery v1 — recipe-based (instances regenerated
deterministically; D matrices never persisted).

Tier 1: spheres, hyperbolic balls, flat tori, mixed products, d in 2-6.
Tier 2: dumbbell warped products, d in 3-6, beta in {0.5, 0.8}.
Noise axis: multiplicative distance noise sigma in {0, 0.05, 0.15}.
3 seeds per cell; n = 1500.

`materialize(recipe)` -> dict(D, X, ks_field, eval_idx, **recipe).
Warped-product 2D tables are cached in-process per (profile, d).

Usage: pixi run python build_battery.py   (writes recipes joblib + counts)
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           hyperbolic, necklace_profile,
                                           product, sphere_sf, torus_flat)

N = 1500
SEEDS = (0, 1, 2)
NOISES = (0.0, 0.05, 0.15)
OUT = Path("processed_data/menagerie_recipes.joblib")


def tier1_recipes():
    rec = []
    for d in (2, 3, 4, 5, 6):
        for r in (0.7, 1.0, 1.5):
            rec.append(dict(kind="sphere", dim=d, r=r))
        for kappa in (0.5, 1.0, 2.0):
            rec.append(dict(kind="hyperbolic", dim=d, kappa=kappa))
        rec.append(dict(kind="torus", dim=d))
    for (a, ka), (b, kb) in [((2, None), (2, 1.0)), ((2, None), (3, 1.0)),
                             ((3, None), (2, 0.5)), ((3, None), (3, 2.0)),
                             ((2, None), (4, 1.0))]:
        rec.append(dict(kind="product_SH", da=a, db=b, kappa=kb,
                        dim=a + b))
    return rec


def tier2_recipes():
    rec = [dict(kind="dumbbell", dim=d, beta=beta)
           for d, beta in itertools.product((3, 4, 5, 6), (0.5, 0.8))]
    rec += [dict(kind="necklace", dim=d, b=b)
            for d, b in itertools.product((3, 4, 5, 6), (0.55, 0.7))]
    return rec


@lru_cache(maxsize=8)
def _wp(beta: float, d: int) -> WarpedProduct:
    f, L = dumbbell_profile(beta=beta)
    return WarpedProduct(f, L, d=d)


@lru_cache(maxsize=8)
def _wp_neck(b: float, d: int) -> WarpedProduct:
    f, L = necklace_profile(b=b)
    return WarpedProduct(f, L, d=d, periodic=True)


def materialize(recipe: dict) -> dict:
    seed, noise = recipe["seed"], recipe["noise"]
    rng = np.random.default_rng(seed * 7919 + recipe["dim"])
    k = recipe["kind"]
    if k == "sphere":
        m = sphere_sf(N, recipe["dim"], r=recipe["r"], rng=rng)
    elif k == "hyperbolic":
        m = hyperbolic(N, recipe["dim"], kappa=recipe["kappa"], rng=rng)
    elif k == "torus":
        m = torus_flat(N, recipe["dim"], rng=rng)
    elif k == "product_SH":
        m = product(sphere_sf(N, recipe["da"], r=1.0, rng=rng),
                    hyperbolic(N, recipe["db"], kappa=recipe["kappa"],
                               rng=rng))
    elif k == "dumbbell":
        m = _wp(recipe["beta"], recipe["dim"]).sample(N, rng=rng)
    elif k == "necklace":
        m = _wp_neck(recipe["b"], recipe["dim"]).sample(N, rng=rng)
    else:
        raise ValueError(k)
    D = m["D"]
    # scale normalization: unit median geodesic distance; curvature in
    # data-scale units (K -> K * s^2). Makes cross-manifold comparison fair.
    scale = float(np.median(D[np.triu_indices_from(D, k=1)]))
    D = D / scale
    ks_field = np.asarray(m["ks_field"], dtype=float) * scale**2
    if noise > 0:
        nz = np.random.default_rng(seed + 5000)
        eps = nz.normal(scale=noise, size=D.shape)
        eps = 0.5 * (eps + eps.T)
        D = np.maximum(D * (1 + eps), 0)
        np.fill_diagonal(D, 0)
    n_eval = 24 if k in ("dumbbell", "necklace") else 8
    if k in ("dumbbell", "necklace"):  # stratify eval points by ks quantile
        qs = np.linspace(0.02, 0.98, n_eval)
        order = np.argsort(ks_field)
        eval_idx = order[(qs * (len(order) - 1)).astype(int)]
    else:
        eval_idx = np.random.default_rng(seed).choice(N, n_eval,
                                                      replace=False)
    return dict(D=D, X=None, ks_field=ks_field, scale=scale,
                eval_idx=np.asarray(eval_idx), name=m["meta"]["name"],
                **recipe)


def main() -> None:
    recipes = []
    for base in tier1_recipes() + tier2_recipes():
        for seed, noise in itertools.product(SEEDS, NOISES):
            recipes.append(dict(**base, seed=seed, noise=noise,
                                dataset=("tier2" if base["kind"] in ("dumbbell", "necklace")
                                         else "tier1")))
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(recipes, OUT)
    print(f"wrote {OUT}: {len(recipes)} instances "
          f"({sum(r['dataset']=='tier1' for r in recipes)} tier1, "
          f"{sum(r['dataset']=='tier2' for r in recipes)} tier2)")


if __name__ == "__main__":
    main()
