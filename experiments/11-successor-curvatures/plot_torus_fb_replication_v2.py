"""Plot v2 torus FB sweep: best-over-τ correlation vs N, faceted by γ, one line per T.

Also a heatmap of best τ across (γ, N) at T=50.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _best_over_temp(df: pd.DataFrame, corr_col: str) -> pd.DataFrame:
    """For each (seed, T, N, γ), pick the τ that maximizes `corr_col`."""
    df = df.copy()
    df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
    idx = df.groupby(["seed", "T", "N", "gamma"])[corr_col].idxmax()
    idx = idx.dropna().astype(int)
    return df.loc[idx].reset_index(drop=True)


def _agg(best: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for (T, N, gamma), g in best.groupby(["T", "N", "gamma"]):
        v = g[value_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append(dict(
            T=int(T), N=int(N), gamma=float(gamma),
            mean=float(np.mean(v)) if v.size else float("nan"),
            std=float(np.std(v)) if v.size > 1 else 0.0,
            n=len(g),
        ))
    return pd.DataFrame(rows).sort_values(["gamma", "T", "N"])


def _plot_facet(
    best: pd.DataFrame, value_col: str, out_stem: Path,
    ylabel: str, title: str, yline: float | None = None,
    ylim: tuple[float, float] | None = None, target: float | None = None,
) -> None:
    agg = _agg(best, value_col)
    gammas = sorted(agg["gamma"].unique())
    Ts = sorted(agg["T"].unique())

    sns.set_context("talk")
    fig, axes = plt.subplots(
        1, len(gammas), figsize=(4.5 * len(gammas), 5.0),
        sharey=True, squeeze=False,
    )
    palette = sns.color_palette("rocket_r", len(Ts))

    for i, gamma in enumerate(gammas):
        ax = axes[0, i]
        for T, color in zip(Ts, palette):
            sub = agg[(agg.gamma == gamma) & (agg.T == T)]
            ax.errorbar(
                sub["N"], sub["mean"], yerr=sub["std"],
                marker="o", capsize=3, linewidth=2,
                color=color, label=f"T = {T}",
            )
        if yline is not None:
            ax.axhline(yline, color="gray", ls="--", lw=0.8, alpha=0.6)
        if target is not None:
            ax.axhline(target, color="green", ls=":", lw=1.2, alpha=0.7,
                       label=f"target ({target})")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("N walks")
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(f"γ = {gamma}", fontsize=13)
        ax.grid(alpha=0.3)
        if i == len(gammas) - 1:
            ax.legend(loc="best", fontsize=11)

    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def _plot_best_temp_heatmap(best: pd.DataFrame, out_stem: Path) -> None:
    """Heatmap of the optimal τ by (γ, N) averaged over seeds, separate for each T."""
    Ts = sorted(best["T"].unique())
    sns.set_context("talk")
    fig, axes = plt.subplots(1, len(Ts), figsize=(5.2 * len(Ts), 4.5), squeeze=False)
    for ax, T in zip(axes[0], Ts):
        sub = best[best["T"] == T]
        grid = sub.pivot_table(index="gamma", columns="N",
                               values="temperature", aggfunc="mean")
        sns.heatmap(grid, ax=ax, cmap="magma", annot=True, fmt=".2f",
                    cbar_kws={"label": "best τ"})
        ax.set_title(f"T = {T}")
        ax.set_xlabel("N walks")
        ax.set_ylabel("γ")
    fig.suptitle("Torus FB v2: optimal softmax τ per config (mean over seeds)",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/torus_fb_replication_v2.csv")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    out_dir = Path(args.out_dir)

    # Best over τ, keyed by Spearman (main signal of interest).
    best_sp = _best_over_temp(df, "spearman_entropy_vs_ks")
    best_pe = _best_over_temp(df, "pearson_entropy_vs_ks")
    # KL: smallest KL per config (independent of which τ maxes correlation).
    df["_neg_kl"] = -df["mean_kl"]
    best_kl = _best_over_temp(df, "_neg_kl")

    _plot_facet(
        best_sp, "spearman_entropy_vs_ks",
        out_stem=out_dir / "torus_fb_v2_spearman_best",
        ylabel="Spearman(H(P̂), ks) — best over τ",
        title="Torus FB v2: best-temperature Spearman vs N (faceted by γ)",
        yline=0.0, ylim=(-0.1, 0.75), target=0.5,
    )
    _plot_facet(
        best_pe, "pearson_entropy_vs_ks",
        out_stem=out_dir / "torus_fb_v2_pearson_best",
        ylabel="Pearson(H(P̂), ks) — best over τ",
        title="Torus FB v2: best-temperature Pearson vs N (faceted by γ)",
        yline=0.0, ylim=(-0.1, 0.75), target=0.5,
    )
    _plot_facet(
        best_kl, "mean_kl",
        out_stem=out_dir / "torus_fb_v2_kl_best",
        ylabel="min row-KL(real ‖ softmax(M/τ))",
        title="Torus FB v2: min over τ of reconstruction KL vs N (faceted by γ)",
    )
    _plot_best_temp_heatmap(best_sp, out_dir / "torus_fb_v2_best_temp_heatmap")


if __name__ == "__main__":
    main()
