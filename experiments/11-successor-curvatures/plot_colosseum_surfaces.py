"""Multi-panel visualization of representative 2-surfaces from the Curvature Colosseum.

Generates a pool of random d=2, N=3, degree=2 surfaces and renders surfaces
spanning the curvature range as a smooth mesh with the rejection-sampled point
cloud overlaid.

Layouts:
- `horizontal` (default): 3 panels in a row — strong negative, near-flat, strong positive.
- `square`: 4x4 grid — rows are four surfaces spanning the curvature spectrum
  (strong−, mild−, mild+, strong+), columns are the v2-benchmark noise levels
  ε ∈ {0.01, 0.05, 0.1, 0.2}.

Outputs:
- figures/<out_stem>.svg
- figures/<out_stem>.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import cm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from diffusion_curvature.random_surfaces import (  # noqa: E402
    random_surface,
    rejection_sample_from_surface,
    scalar_curvature_at_origin,
)


NOISE_LEVELS_V2 = (0.01, 0.05, 0.10, 0.20)


def generate_surface(seed: int, degree: int = 2, noise: float = 0.0):
    """Generate a random 2-surface; sample n_points and optionally add ambient noise."""
    np.random.seed(seed)
    F, _ = random_surface(d=2, N=3, degree=degree)
    k = float(scalar_curvature_at_origin(F))
    X = rejection_sample_from_surface(F, n_points=1500, seed=seed)
    if noise > 0:
        rng = np.random.default_rng(seed + 9973)
        X = X + rng.normal(scale=noise, size=X.shape)
    return F, k, X


def evaluate_surface_on_grid(F, n_grid: int = 80, bounds=(-1.0, 1.0)):
    vars_ = sorted(F.free_symbols, key=lambda v: v.name)
    z_expr = F[2]
    f_np = sp.lambdify(vars_, z_expr, "numpy")
    xs = np.linspace(bounds[0], bounds[1], n_grid)
    ys = np.linspace(bounds[0], bounds[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = f_np(XX, YY)
    if np.isscalar(ZZ) or (hasattr(ZZ, "ndim") and ZZ.ndim == 0):
        ZZ = np.full_like(XX, float(ZZ))
    return XX, YY, ZZ


def pick_three(pool: list[tuple[int, float]]) -> list[tuple[int, float]]:
    pool_sorted = sorted(pool, key=lambda t: t[1])
    neg = pool_sorted[0]
    pos = pool_sorted[-1]
    near_zero = min(pool, key=lambda t: abs(t[1]))
    chosen: list[tuple[int, float]] = []
    for item in [neg, near_zero, pos]:
        if item not in chosen:
            chosen.append(item)
    return chosen


def pick_four(pool: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Pick four seeds: strong neg, mild neg, mild pos, strong pos (sorted by curvature)."""
    pool_sorted = sorted(pool, key=lambda t: t[1])
    negatives = [t for t in pool_sorted if t[1] < 0]
    positives = [t for t in pool_sorted if t[1] >= 0]
    if not negatives or not positives:
        # Fall back to four quartile picks if the pool is one-sided.
        idxs = [0, len(pool_sorted) // 3, 2 * len(pool_sorted) // 3, len(pool_sorted) - 1]
        return [pool_sorted[i] for i in idxs]
    strong_neg = negatives[0]
    mild_neg = negatives[-1]
    mild_pos = positives[0]
    strong_pos = positives[-1]
    chosen: list[tuple[int, float]] = []
    for item in [strong_neg, mild_neg, mild_pos, strong_pos]:
        if item not in chosen:
            chosen.append(item)
    for item in pool_sorted:
        if len(chosen) >= 4:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen[:4]


def _draw_panel(ax, F, X, point_kwargs: dict, surface_alpha: float):
    XX, YY, ZZ = evaluate_surface_on_grid(F, n_grid=80)
    ax.plot_surface(
        XX, YY, ZZ,
        cmap=cm.viridis,
        alpha=surface_alpha,
        linewidth=0,
        antialiased=True,
        rstride=2, cstride=2,
    )
    mask = (np.abs(X[:, 0]) <= 1.05) & (np.abs(X[:, 1]) <= 1.05)
    ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], **point_kwargs)
    ax.set_xlabel("$x_0$", fontsize=8)
    ax.set_ylabel("$x_1$", fontsize=8)
    ax.set_zlabel("$z$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=25, azim=-60)


def render(
    out_stem: Path,
    seeds: list[int],
    degree: int = 2,
    layout: str = "horizontal",
):
    pool: list[tuple[int, float]] = []
    cache: dict[int, tuple] = {}
    for seed in seeds:
        F, k, _ = generate_surface(seed, degree=degree, noise=0.0)
        pool.append((seed, k))
        cache[seed] = (F, k)
        print(f"  seed={seed:3d}  k(origin)={k:+.3f}")

    if layout == "square":
        chosen = pick_four(pool)
        labels = [
            "Strong negative",
            "Mild negative",
            "Mild positive",
            "Strong positive",
        ]
        # Pair the four curvature picks with the four v2 noise levels.
        noise_levels = NOISE_LEVELS_V2
        print("\nChosen surfaces (curvature × noise pairing):")
        for (seed, k), noise in zip(chosen, noise_levels):
            print(f"  seed={seed}  k={k:+.3f}  noise={noise:g}")

        light_points = dict(
            s=6, color="#3a3a3a", edgecolor="white",
            linewidth=0.2, alpha=0.65, depthshade=False,
        )

        fig = plt.figure(figsize=(11, 10))
        for i, ((seed, k), label, noise) in enumerate(
            zip(chosen, labels, noise_levels)
        ):
            F, _ = cache[seed]
            _, _, X_noisy = generate_surface(seed, degree=degree, noise=noise)
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            _draw_panel(ax, F, X_noisy, light_points, surface_alpha=0.55)
            ax.set_title(
                f"{label} ($k_{{\\mathrm{{origin}}}} = {k:+.2f}$),  "
                f"noise $\\varepsilon = {noise:g}$",
                fontsize=12, pad=6,
            )

        fig.suptitle(
            "Four 2-surfaces from the Curvature Colosseum — varied curvature & noise "
            "(d=2, codim=1, degree=2)",
            fontsize=14, y=1.0,
        )
        fig.tight_layout()
    else:
        chosen = pick_three(pool)
        titles = ["Negative curvature", "Near-flat", "Positive curvature"]
        print("\nChosen surfaces (horizontal):")
        for seed, k in chosen:
            print(f"  seed={seed}  k={k:+.3f}")

        strong_points = dict(
            s=8, color="#1a1a1a", edgecolor="white",
            linewidth=0.25, alpha=0.9, depthshade=False,
        )
        fig = plt.figure(figsize=(15, 5))
        for i, ((seed, k), title) in enumerate(zip(chosen, titles)):
            F, _ = cache[seed]
            _, _, X = generate_surface(seed, degree=degree, noise=0.0)
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            _draw_panel(ax, F, X, strong_points, surface_alpha=0.45)
            ax.set_title(f"{title}\n$k_{{\\mathrm{{origin}}}} = {k:+.2f}$", fontsize=13)
        fig.suptitle(
            "Representative 2-surfaces from the Curvature Colosseum "
            "(d=2, codim=1, degree=2)",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"\nSaved: {out_stem}.svg and {out_stem}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/colosseum_surfaces")
    parser.add_argument("--n-pool", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument(
        "--layout", choices=["horizontal", "square"], default="horizontal"
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_pool))
    out_stem = Path(args.out)
    render(out_stem, seeds, degree=args.degree, layout=args.layout)


if __name__ == "__main__":
    main()
