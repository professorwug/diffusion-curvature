"""Plot torus FB replication sweep: KL and entropy-curvature correlation vs N,
faceted by γ, one line per T.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _agg(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for (T, N, gamma), g in df.groupby(["T", "N", "gamma"]):
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
    df: pd.DataFrame, value_col: str, out_stem: Path,
    ylabel: str, title: str, yline: float | None = None,
    ylim: tuple[float, float] | None = None, logy: bool = False,
) -> None:
    agg = _agg(df, value_col)
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
        if logy:
            ax.set_yscale("log")
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/torus_fb_replication.csv")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    out_dir = Path(args.out_dir)

    _plot_facet(
        df, "mean_kl",
        out_stem=out_dir / "torus_fb_kl",
        ylabel="mean row KL(real ‖ reconstructed)",
        title="Torus FB: reconstruction KL vs N walks (faceted by γ)",
        logy=False,
    )
    _plot_facet(
        df, "pearson_entropy_vs_ks",
        out_stem=out_dir / "torus_fb_pearson_entropy_ks",
        ylabel="Pearson(H(P̂), ks)",
        title="Torus FB: row-entropy ↔ Gaussian curvature (faceted by γ)",
        yline=0.0, ylim=(-1.05, 1.05),
    )
    _plot_facet(
        df, "spearman_entropy_vs_ks",
        out_stem=out_dir / "torus_fb_spearman_entropy_ks",
        ylabel="Spearman(H(P̂), ks)",
        title="Torus FB: row-entropy ↔ Gaussian curvature, Spearman (faceted by γ)",
        yline=0.0, ylim=(-1.05, 1.05),
    )
    _plot_facet(
        df, "pearson_entropy_hat_vs_real",
        out_stem=out_dir / "torus_fb_pearson_entropy_hat_vs_real",
        ylabel="Pearson(H(P̂), H(P^(γ)))",
        title="Torus FB: reconstructed row-entropy vs real row-entropy (faceted by γ)",
        yline=0.0, ylim=(-1.05, 1.05),
    )


if __name__ == "__main__":
    main()
