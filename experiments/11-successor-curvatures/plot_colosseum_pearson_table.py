r"""Curvature Colosseum Pearson r as a 'classic table'.

Layout differs from the merged figure (`plot_colosseum_pearson_merged.py`):

    rows    = methods (one per row, left-labelled)
    columns = (intrinsic dim, noise level), grouped by dim
    cell    = diagonal-split square, upper-left = iid Pearson r,
                                     lower-right = τ-sampled Pearson r

The point is that the previous "method block × dim panel" layout reads as
a heatmap *per method*, which confuses readers who are used to a single
table-of-numbers presentation. This re-cast makes the figure unambiguously
a table — methods stacked vertically, conditions running horizontally.

SPC is excluded (only partial coverage at d≤4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

# Half-width column placement → font scale tuned so the default tick
# numbers stay readable at ~14.5 pt. Individual title/label sizes are
# overridden below to ≥18 pt.
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.45)


# Drop SPC. Order from "winning method" downward.
FAMILIES: list[tuple[str, list[str]]] = [
    ("Successor", ["successor_entropy"]),
    ("DC",        ["dc2", "dc_wasserstein_ollivier", "dc_entropic_subtraction"]),
    ("Hickock",   ["hickock"]),
    ("ORC",       ["orc"]),
    ("FRC",       ["frc_weighted_adaptive", "frc_unweighted_knn"]),
]

DIMS = [2, 3, 4, 5, 6]
NOISES = [0.01, 0.05, 0.10, 0.20]

# Visual knobs.
GROUP_GAP = 0.30   # gap (in cell-widths) between dim groups
NUMERIC_FONT = 6.5


# ---------------------------------------------------------------------------
# Data plumbing (parallel to the merged-figure script)
# ---------------------------------------------------------------------------


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def build_summary_from_metrics(metrics_csv: Path,
                               drop_methods: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if drop_methods:
        df = df[~df["method"].isin(drop_methods)]
    rows = []
    for (method, d, noise), g in df.groupby(["method", "dim", "noise"]):
        rows.append(dict(
            method=method, dim=int(d), noise=float(noise),
            pearson=_pearson(g["ks_hat"], g["ks_true"]),
            n=len(g),
        ))
    return pd.DataFrame(rows)


def select_variant(summary: pd.DataFrame, candidates: list[str]) -> str | None:
    sub = summary[summary.method.isin(candidates)]
    if sub.empty:
        return None
    means = sub.groupby("method")["pearson"].mean()
    return means.sort_values(ascending=False).index[0]


def lookup(summary: pd.DataFrame, method: str | None, dim: int, noise: float) -> float:
    if not method:
        return float("nan")
    m = summary[
        (summary.method == method)
        & (summary.dim == dim)
        & np.isclose(summary.noise, noise)
    ]
    if not len(m):
        return float("nan")
    return float(m.pearson.iloc[0])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(iid_summary: pd.DataFrame, tau_summary: pd.DataFrame,
                out_stem: Path) -> None:
    family_method: list[tuple[str, str | None]] = []
    for family, candidates in FAMILIES:
        m = select_variant(iid_summary, candidates) or select_variant(tau_summary, candidates)
        family_method.append((family, m))

    n_methods = len(family_method)
    n_dims = len(DIMS)
    n_noises = len(NOISES)

    cell_w = cell_h = 1.0
    # Column x-centers, with gaps between dim groups but none within groups.
    x_centers: list[float] = []
    group_centers: list[float] = []
    group_left: list[float] = []
    group_right: list[float] = []
    x = 0.0
    for di in range(n_dims):
        if di > 0:
            x += GROUP_GAP
        block_left = x
        for ni in range(n_noises):
            x_centers.append(x + cell_w / 2)
            x += cell_w
        block_right = x
        group_left.append(block_left)
        group_right.append(block_right)
        group_centers.append((block_left + block_right) / 2)
    total_w = x

    # Row y-centers — top row is the first method (Successor at the top).
    y_centers = [(n_methods - 1 - mi) + 0.5 for mi in range(n_methods)]
    total_h = n_methods * cell_h

    fig_w = 0.65 * total_w + 3.2
    fig_h = 0.55 * total_h + 4.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # RdYlGn anchors red at -1, yellow at 0, green at +1 — so high Pearson
    # ρ reads as 'green = good'.
    cmap = mpl.colormaps["RdYlGn"]
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

    for mi, (family, method) in enumerate(family_method):
        yc = y_centers[mi]
        for di, d in enumerate(DIMS):
            for ni, noise in enumerate(NOISES):
                col_idx = di * n_noises + ni
                xc = x_centers[col_idx]
                cx0, cx1 = xc - cell_w / 2, xc + cell_w / 2
                cy0, cy1 = yc - cell_h / 2, yc + cell_h / 2

                v_iid = lookup(iid_summary, method, d, noise)
                v_tau = lookup(tau_summary, method, d, noise)

                upper = mpatches.Polygon(
                    [(cx0, cy1), (cx1, cy1), (cx0, cy0)],
                    closed=True,
                    facecolor=cmap(norm(v_iid)) if np.isfinite(v_iid) else "#dddddd",
                    edgecolor="white", linewidth=0.5,
                )
                ax.add_patch(upper)
                lower = mpatches.Polygon(
                    [(cx1, cy1), (cx1, cy0), (cx0, cy0)],
                    closed=True,
                    facecolor=cmap(norm(v_tau)) if np.isfinite(v_tau) else "#dddddd",
                    edgecolor="white", linewidth=0.5,
                )
                ax.add_patch(lower)

                def _tc(v):
                    return "white" if (np.isfinite(v) and abs(v) > 0.55) else "black"

                if np.isfinite(v_iid):
                    ax.text(
                        cx0 + 0.30 * cell_w, cy0 + 0.74 * cell_h,
                        f"{v_iid:.2f}",
                        ha="center", va="center",
                        fontsize=NUMERIC_FONT, color=_tc(v_iid),
                    )
                if np.isfinite(v_tau):
                    ax.text(
                        cx0 + 0.70 * cell_w, cy0 + 0.26 * cell_h,
                        f"{v_tau:.2f}",
                        ha="center", va="center",
                        fontsize=NUMERIC_FONT, color=_tc(v_tau),
                    )

    # Bounds.
    ax.set_xlim(-0.05, total_w + 0.05)
    ax.set_ylim(-0.05, total_h + 0.05)
    ax.set_aspect("equal")

    # Method labels along the left.
    ax.set_yticks(y_centers)
    ax.set_yticklabels([fam for fam, _ in family_method], fontsize=18,
                       fontweight="bold")
    ax.tick_params(axis="y", which="both", length=0)

    # Inner column ticks: noise level inside each dim group.
    inner_labels = [f"{n:g}" for n in NOISES] * n_dims
    ax.set_xticks(x_centers)
    ax.set_xticklabels(inner_labels, fontsize=12)
    ax.tick_params(axis="x", which="both", length=0, pad=2)

    # Group headers: "N-Manifolds" centred above each dim block.
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for d, center in zip(DIMS, group_centers):
        ax.text(center, 1.02, f"{d}-Manifolds",
                transform=trans, ha="center", va="bottom",
                fontsize=18, fontweight="bold")

    # Hide spines.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Pearson r colorbar (vertical, on the right). The cell colour is the
    # primary signal; the per-triangle numbers act as a secondary readout.
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=25)
    cbar.set_label(r"Pearson $\rho$", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel("noise level ε", fontsize=20, labelpad=10)

    # Diagonal-split legend in its own small axes, anchored to the
    # bottom-LEFT gutter — symmetric with the colorbar on the right and
    # away from the table's columns. A neutral-grey example cell makes the
    # split convention explicit without recalling any data value.
    legend_ax = fig.add_axes([0.04, 0.02, 0.34, 0.10])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    # Square in (0, 0.92) horizontally, sized to ~0.10 wide for a small chip.
    sq_x0, sq_x1 = 0.02, 0.18
    sq_y0, sq_y1 = 0.15, 0.85
    upper = mpatches.Polygon(
        [(sq_x0, sq_y1), (sq_x1, sq_y1), (sq_x0, sq_y0)],
        closed=True, facecolor="#bdbdbd", edgecolor="white", linewidth=0.8,
    )
    lower = mpatches.Polygon(
        [(sq_x1, sq_y1), (sq_x1, sq_y0), (sq_x0, sq_y0)],
        closed=True, facecolor="#7d7d7d", edgecolor="white", linewidth=0.8,
    )
    legend_ax.add_patch(upper)
    legend_ax.add_patch(lower)
    legend_ax.text(sq_x1 + 0.03, sq_y1,
                    "uniformly-sampled",
                    ha="left", va="top", fontsize=13, color="#222")
    legend_ax.text(sq_x1 + 0.03, sq_y0,
                    "trajectory-sampled",
                    ha="left", va="bottom", fontsize=13, color="#222")

    fig.suptitle(
        r"The 'Curvature Colosseum': Recovery of Scalar Curvature from "
        r"noisy $d$-Surfaces",
        fontsize=22, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.905,
        r"Successor Entropy has higher Pearson $\rho$ than existing "
        r"methods in high dimensions",
        ha="center", va="top", fontsize=20, color="#444",
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.09, right=0.95)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,pdf,png}}")
    print("Family → variant chosen:")
    for fam, m in family_method:
        print(f"  {fam:>10s}  ←  {m}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iid-summary",
                   default="processed_data/v2/colosseum_pearson_summary_v3.csv")
    p.add_argument("--tau-summary",
                   default="processed_data/v2/tau_colosseum_pearson_summary.csv")
    p.add_argument("--tau-metrics",
                   default="processed_data/tau_metrics.csv",
                   help="Used as a fallback if --tau-summary is not present.")
    p.add_argument("--include-tau-successor", action="store_true",
                   help="Don't mask successor_* in the τ summary (use only "
                   "after the FB rerun has landed).")
    p.add_argument("--out", default="figures/v2/colosseum_pearson_table")
    args = p.parse_args()

    iid = pd.read_csv(args.iid_summary)

    tau_summary_path = Path(args.tau_summary)
    if tau_summary_path.exists():
        tau = pd.read_csv(tau_summary_path)
    else:
        drop = () if args.include_tau_successor else ("successor_entropy", "successor_orc")
        tau = build_summary_from_metrics(Path(args.tau_metrics), drop_methods=drop)

    make_figure(iid, tau, Path(args.out))


if __name__ == "__main__":
    main()
