r"""Merged, rotated, diagonal-split Colosseum Pearson heatmap.

Combines the IID and trajectory-sampled (TauColosseum) benchmarks in a single
figure. Each (method × dim × ε) cell is drawn as a square split along its
main diagonal:

    ┌───────┐
    │ \  i  │     upper-left triangle  = IID Pearson r
    │  \    │
    │ τ  \  │     lower-right triangle = trajectory-sampled Pearson r
    └───────┘

The figure is rotated relative to the original compact panel:

    x-axis  =  methods (grouped) × ε within each method
    y-axis  =  intrinsic dimension d ∈ {2..6}

This aspect fits a single column of a NeurIPS page far more comfortably
than the previous tall stack.

Data sources (defaults):
    --iid-summary   processed_data/v2/colosseum_pearson_summary.csv
                    (the v2 / pre-today FB config — preserves the salvaged
                    successor-entropy values the user wants to keep until
                    the cluster rerun completes.)
    --tau-summary   built on the fly from processed_data/tau_metrics.csv,
                    masking out successor_* (those still need the FB rerun).
                    Pass an explicit summary CSV via --tau-summary to override.
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


FAMILIES: list[tuple[str, list[str]]] = [
    ("Successor", ["successor_entropy"]),
    ("DC",        ["dc2", "dc_wasserstein_ollivier", "dc_entropic_subtraction"]),
    ("Hickock",   ["hickock"]),
    ("SPC",       ["spc_hop", "spc_diffusion_t5"]),
    ("ORC",       ["orc"]),
    ("FRC",       ["frc_weighted_adaptive", "frc_unweighted_knn"]),
]

DIMS = [2, 3, 4, 5, 6]
NOISES = [0.01, 0.05, 0.10, 0.20]

# Visual layout knobs.
NOISE_GAP = 0.0    # gap between noise columns within a method block
METHOD_GAP = 0.45  # gap between method blocks
DIM_GAP = 0.0


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
    """Recompute (method, dim, noise) Pearson summary from a metrics CSV."""
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


def lookup(summary: pd.DataFrame, method: str, dim: int, noise: float) -> float:
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


def make_figure(iid_summary: pd.DataFrame, tau_summary: pd.DataFrame,
                out_stem: Path) -> None:
    # Resolve which method variant to use per family. Prefer iid; fall back
    # to tau if a method has no iid coverage (it shouldn't, but be safe).
    family_method: list[tuple[str, str | None]] = []
    for family, candidates in FAMILIES:
        m = select_variant(iid_summary, candidates) or select_variant(tau_summary, candidates)
        family_method.append((family, m))

    n_families = len(family_method)
    n_noises = len(NOISES)
    n_dims = len(DIMS)

    # X-axis layout: 6 method blocks of 4 noise cells each, with gaps.
    cell_w = 1.0
    cell_h = 1.0
    x_centers: list[float] = []
    fam_centers: list[float] = []
    fam_left: list[float] = []
    fam_right: list[float] = []
    x = 0.0
    for fi, (_, _) in enumerate(family_method):
        if fi > 0:
            x += METHOD_GAP
        block_x_centers = []
        block_left = x
        for ni, _ in enumerate(NOISES):
            if ni > 0:
                x += NOISE_GAP
            block_x_centers.append(x + cell_w / 2)
            x += cell_w
        x_centers.extend(block_x_centers)
        fam_left.append(block_left)
        fam_right.append(x)
        fam_centers.append((block_left + x) / 2)
    total_w = x

    # Y-axis: 5 dimension rows. Higher d at top, lower d at bottom.
    # We invert the y-axis at the end so the smallest dim is at the bottom.
    y_centers = [(n_dims - 1 - di) + 0.5 for di in range(n_dims)]
    total_h = n_dims * cell_h

    # Figure size: aim for column-width × ~half height. We bias slightly
    # taller so the per-cell numeric annotations remain legible.
    fig_w = 0.40 * total_w + 1.4
    fig_h = 0.65 * total_h + 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = mpl.colormaps["RdBu_r"]
    cmap = cmap.with_extremes(bad="#dddddd")  # NaN cells = neutral grey
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

    # Iterate cells: for each method × noise (column j) × dim (row i), draw
    # two triangles encoding the iid and tau Pearson r.
    for fi, (family, method) in enumerate(family_method):
        for ni, noise in enumerate(NOISES):
            j = fi * n_noises + ni
            xc = x_centers[j]
            for di, d in enumerate(DIMS):
                yc = y_centers[di]
                cx0, cx1 = xc - cell_w / 2, xc + cell_w / 2
                cy0, cy1 = yc - cell_h / 2, yc + cell_h / 2

                v_iid = lookup(iid_summary, method, d, noise)
                v_tau = lookup(tau_summary, method, d, noise)

                # Upper-left triangle = IID, with vertices
                #   (cx0, cy1)       top-left
                #   (cx1, cy1)       top-right
                #   (cx0, cy0)       bottom-left
                upper = mpatches.Polygon(
                    [(cx0, cy1), (cx1, cy1), (cx0, cy0)],
                    closed=True,
                    facecolor=cmap(norm(v_iid)) if np.isfinite(v_iid) else "#dddddd",
                    edgecolor="white", linewidth=0.5,
                )
                ax.add_patch(upper)

                # Lower-right triangle = τ
                lower = mpatches.Polygon(
                    [(cx1, cy1), (cx1, cy0), (cx0, cy0)],
                    closed=True,
                    facecolor=cmap(norm(v_tau)) if np.isfinite(v_tau) else "#dddddd",
                    edgecolor="white", linewidth=0.5,
                )
                ax.add_patch(lower)

                # Optional value annotations: small text in each triangle.
                # Skip if the cell is too small to read; here we keep them.
                def _text_color(v):
                    return "white" if (np.isfinite(v) and abs(v) > 0.55) else "black"

                if np.isfinite(v_iid):
                    ax.text(
                        cx0 + 0.32 * cell_w, cy0 + 0.74 * cell_h,
                        f"{v_iid:.2f}",
                        ha="center", va="center",
                        fontsize=6.5, color=_text_color(v_iid),
                    )
                if np.isfinite(v_tau):
                    ax.text(
                        cx0 + 0.68 * cell_w, cy0 + 0.26 * cell_h,
                        f"{v_tau:.2f}",
                        ha="center", va="center",
                        fontsize=6.5, color=_text_color(v_tau),
                    )

    # Axes setup.
    ax.set_xlim(-0.05, total_w + 0.05)
    ax.set_ylim(-0.05, total_h + 0.05)
    ax.set_aspect("equal")

    # Y ticks at dim row centers.
    ax.set_yticks(y_centers)
    ax.set_yticklabels([f"d = {d}" for d in DIMS], fontsize=9)
    ax.set_ylabel("intrinsic dimension", fontsize=10, labelpad=6)

    # X ticks at noise centers, labelled with noise only.
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"{n:.2g}" for n in NOISES] * n_families, fontsize=7)
    ax.tick_params(axis="x", which="both", length=0, pad=2)

    # Family labels above their blocks (just above the heatmap; the title
    # sits higher still).
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for center, (family, _) in zip(fam_centers, family_method):
        ax.text(center, 1.015, family,
                transform=trans, ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")

    # Noise label on the bottom axis. The diagonal-split convention is
    # spelled out inline so we don't need a separate legend patch.
    ax.set_xlabel(
        "noise level ε       "
        "values: Pearson ρ on iid samples / trajectory-sampled",
        fontsize=10, labelpad=10,
    )

    # Hide spines / tick marks for a cleaner heatmap look.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", which="both", length=0)

    fig.suptitle(
        "Curvature recovery on the Colosseum benchmark — iid vs τ-sampled",
        fontsize=11, y=0.97,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    # Manual margin control — tight_layout fights with the suptitle and the
    # absolute-positioned legend, leaving an awkward dead band above the axes.
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.98)
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
                   default="processed_data/v2/colosseum_pearson_summary.csv",
                   help="Pearson summary for the iid Colosseum (defaults to v2 — "
                   "preserves salvaged successor-entropy values).")
    p.add_argument("--tau-metrics",
                   default="processed_data/tau_metrics.csv",
                   help="Per-instance τ-Colosseum metrics; summary is built on "
                   "the fly. Successor methods are MASKED so they appear blank "
                   "until the FB-rerun finishes.")
    p.add_argument("--tau-summary", default=None,
                   help="If provided, use this τ summary directly instead of "
                   "rebuilding from --tau-metrics.")
    p.add_argument("--out", default="figures/v2/colosseum_pearson_merged")
    p.add_argument("--include-tau-successor", action="store_true",
                   help="Don't mask successor_* in the τ summary (use only "
                   "after the FB rerun has landed).")
    args = p.parse_args()

    iid = pd.read_csv(args.iid_summary)
    if args.tau_summary:
        tau = pd.read_csv(args.tau_summary)
    else:
        drop = () if args.include_tau_successor else ("successor_entropy", "successor_orc")
        tau = build_summary_from_metrics(Path(args.tau_metrics), drop_methods=drop)

    make_figure(iid, tau, Path(args.out))


if __name__ == "__main__":
    main()
