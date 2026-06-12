"""Colosseum Pearson matrix: one heatmap, methods × (dim, noise).

A cleaner visualization than the current faceted panel grid:
- One row per method, ordered by overall mean Pearson (desc).
- One column per (intrinsic_dim, noise) combination.
- Cell = Pearson(ks_hat, ks_true) across manifolds in that slice.
- Diverging colormap centered at 0.

Scales naturally to more dims: as new (d, noise) cells are benchmarked, new
columns appear to the right. Right now only d=2 is present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


METHOD_LABEL = {
    "dc2": "DC2",
    "dc_wasserstein_ollivier": "DC (W. + Ollivier)",
    "dc_entropic_subtraction": "DC (Entr. + Sub.)",
    "hickock": "Hickock",
    "frc_weighted_adaptive": "FRC (w. adapt.)",
    "frc_unweighted_knn": "FRC (unw. kNN)",
    "orc": "ORC",
    "spc_hop": "SPC hop",
    "spc_diffusion_t5": "SPC diffusion (t=5)",
    "laziness_knn_t5": "Laziness (kNN, t=5)",
    "laziness_adaptive_t5": "Laziness (adapt., t=5)",
    "successor_entropy": "Successor Entropy",
    "successor_orc": "Successor ORC (B)",
}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    cc = df[df.dataset == "colosseum"].copy()
    rows = []
    for (method, d, noise), g in cc.groupby(["method", "dim", "noise"]):
        a = g["ks_hat"].to_numpy(dtype=float)
        b = g["ks_true"].to_numpy(dtype=float)
        rows.append(dict(
            method=method, dim=int(d), noise=float(noise),
            pearson=_pearson(a, b), n=len(g),
        ))
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, out_stem: Path) -> None:
    summary = summary.copy()
    summary["col"] = [f"d={d}, ε={n:g}" for d, n in zip(summary.dim, summary.noise)]

    # Pivot to method × col matrix (ordered by mean |pearson| desc).
    grid = summary.pivot(index="method", columns="col", values="pearson")
    # Stable column order: sort by (dim, noise)
    col_order = (
        summary[["col", "dim", "noise"]]
        .drop_duplicates()
        .sort_values(["dim", "noise"])["col"]
        .tolist()
    )
    grid = grid[col_order]

    # Row order: by absolute mean Pearson across cells (desc)
    mean_abs = grid.abs().mean(axis=1).sort_values(ascending=False)
    grid = grid.loc[mean_abs.index]

    # Pretty row labels
    grid.index = [METHOD_LABEL.get(m, m) for m in grid.index]

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(1.6 * len(grid.columns) + 3, 0.55 * len(grid) + 2))
    sns.heatmap(
        grid, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Pearson r"},
        annot_kws={"size": 12},
    )
    ax.set_title(
        "Curvature Colosseum — Pearson(ks_hat, ks_true) across manifolds\n"
        "rows ordered by |mean Pearson|; columns by increasing dim, noise",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/iid_metrics_v2.csv")
    p.add_argument("--out", default="figures/v2/colosseum_pearson_matrix")
    p.add_argument("--summary-out",
                   default="processed_data/v2/colosseum_pearson_summary.csv")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    summary = build_summary(df)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    plot(summary, Path(args.out))


if __name__ == "__main__":
    main()
