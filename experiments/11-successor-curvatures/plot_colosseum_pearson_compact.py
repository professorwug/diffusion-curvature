"""Compact Colosseum Pearson matrix for the NeurIPS writeup.

Layout: a single heatmap with rows grouped by method family. Each family
occupies four contiguous rows (one per noise level ε ∈ {0.01, 0.05, 0.1, 0.2}).
Columns are intrinsic dimension d ∈ {2,3,4,5,6}. Cells encode Pearson r
between estimated and true scalar curvature across the 50 manifolds in that
(method, dim, noise) cell.

Method families and the variant chosen per family:
  Successor  — successor_entropy (the headline method)
  DC         — best of {dc2, dc_wasserstein_ollivier, dc_entropic_subtraction}
  Hickock    — hickock
  ORC        — orc
  FRC        — best of {frc_weighted_adaptive, frc_unweighted_knn}
  SPC        — best of {spc_hop, spc_diffusion_t5}  (Steinerberger Potential Curvature)

The "best" variant within a family is the one with highest mean Pearson r
across (dim, noise) cells, computed from the same summary CSV the figure
itself reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILIES: list[tuple[str, list[str]]] = [
    ("Successor", ["successor_entropy"]),
    ("DC",        ["dc2", "dc_wasserstein_ollivier", "dc_entropic_subtraction"]),
    ("Hickock",   ["hickock"]),
    ("SPC",       ["spc_hop", "spc_diffusion_t5"]),
    ("ORC",       ["orc"]),
    ("FRC",       ["frc_weighted_adaptive", "frc_unweighted_knn"]),
]

NOISES = [0.01, 0.05, 0.10, 0.20]
DIMS = [2, 3, 4, 5, 6]

# Heights in data units used to build pcolormesh y-edges.
DATA_ROW_HEIGHT = 1.0
SPACER_HEIGHT = 0.22


def select_variant(summary: pd.DataFrame, candidates: list[str]) -> str:
    """Pick the variant from `candidates` with highest mean Pearson r."""
    sub = summary[summary.method.isin(candidates)]
    if sub.empty:
        return candidates[0]
    means = sub.groupby("method")["pearson"].mean()
    return means.sort_values(ascending=False).index[0]


def build_grid(summary: pd.DataFrame) -> tuple[np.ndarray, list[str | None], list[tuple[str, str]], list[int]]:
    """Return (matrix, row_labels, family_blocks, data_row_indices).

    A NaN spacer row is inserted between consecutive method families to
    create true white padding in the heatmap.

    matrix shape: ((n_families * len(NOISES)) + (n_families - 1), len(DIMS))
    row_labels: same length as matrix, with None for spacer rows
    data_row_indices: matrix-row indices of real (non-spacer) rows, in order
    """
    rows = []
    row_labels: list[str | None] = []
    blocks: list[tuple[str, str]] = []
    data_row_indices: list[int] = []
    spacer = [np.nan] * len(DIMS)
    for fam_idx, (family, candidates) in enumerate(FAMILIES):
        if fam_idx > 0:
            rows.append(spacer)
            row_labels.append(None)
        method = select_variant(summary, candidates)
        blocks.append((family, method))
        for eps in NOISES:
            cells = []
            for d in DIMS:
                m = summary[
                    (summary.method == method)
                    & (summary.dim == d)
                    & np.isclose(summary.noise, eps)
                ]
                cells.append(float(m.pearson.iloc[0]) if len(m) else np.nan)
            data_row_indices.append(len(rows))
            rows.append(cells)
            row_labels.append(f"ε = {eps:.2f}")
    return np.asarray(rows, dtype=float), row_labels, blocks, data_row_indices


def plot(grid: np.ndarray, row_labels: list[str | None], blocks: list[tuple[str, str]],
         data_row_indices: list[int], out_stem: Path) -> None:
    n_rows, n_cols = grid.shape
    n_families = len(blocks)
    rows_per_family = len(NOISES)

    # Build y-edges so spacer rows render thin (~SPACER_HEIGHT) while data
    # rows occupy DATA_ROW_HEIGHT in the same data coordinate system.
    is_spacer = [i not in set(data_row_indices) for i in range(n_rows)]
    y_edges = np.zeros(n_rows + 1, dtype=float)
    for i in range(n_rows):
        h = SPACER_HEIGHT if is_spacer[i] else DATA_ROW_HEIGHT
        y_edges[i + 1] = y_edges[i] + h
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    x_edges = np.arange(n_cols + 1, dtype=float)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Total figure height: sum of row heights + chrome.
    total_y = float(y_edges[-1])
    fig, ax = plt.subplots(figsize=(0.85 * n_cols + 2.6, 0.30 * total_y + 1.6))

    cmap = mpl.colormaps["RdBu_r"].with_extremes(bad="white")
    masked = np.ma.masked_invalid(grid)
    im = ax.pcolormesh(x_edges, y_edges, masked,
                       cmap=cmap, vmin=-1, vmax=1, edgecolors="none")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[-1], y_edges[0])  # invert y so first row is at top
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    # Cell annotations only on data rows.
    for i in data_row_indices:
        for j in range(n_cols):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(v) > 0.55 else "black"
            ax.text(x_centers[j], y_centers[i], f"{v:.2f}",
                    ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"d = {d}" for d in DIMS], fontsize=10)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("intrinsic dimension", fontsize=11, labelpad=8)

    ax.set_yticks([y_centers[i] for i in data_row_indices])
    ax.set_yticklabels([row_labels[i] for i in data_row_indices], fontsize=8)

    # Thin white separators between cells inside each method block.
    for j in range(n_cols + 1):
        ax.axvline(x_edges[j], color="white", linewidth=0.3)
    for i in data_row_indices:
        ax.axhline(y_edges[i], color="white", linewidth=0.3)
        ax.axhline(y_edges[i + 1], color="white", linewidth=0.3)

    # Family labels placed left of the noise tick labels.
    family_centers = [
        float(np.mean([y_centers[data_row_indices[k * rows_per_family + r]]
                       for r in range(rows_per_family)]))
        for k in range(n_families)
    ]
    family_names = [b[0] for b in blocks]
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for center, name in zip(family_centers, family_names):
        ax.text(
            -0.22, center, name,
            transform=trans,
            ha="right", va="center",
            fontsize=12, fontweight="bold",
            clip_on=False,
        )

    # Colorbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Pearson r", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Curvature recovery on the Colosseum benchmark",
        fontsize=12, pad=10,
    )

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,pdf,png}}")
    print("Family → variant chosen:")
    for fam, m in blocks:
        print(f"  {fam:>10s}  ←  {m}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary",
        default="processed_data/v2/colosseum_pearson_summary.csv",
        help="Pre-computed Pearson summary (built by plot_colosseum_pearson_matrix.py).",
    )
    p.add_argument("--out", default="figures/v2/colosseum_pearson_compact")
    args = p.parse_args()

    summary = pd.read_csv(args.summary)
    grid, row_labels, blocks, data_row_indices = build_grid(summary)
    plot(grid, row_labels, blocks, data_row_indices, Path(args.out))


if __name__ == "__main__":
    main()
