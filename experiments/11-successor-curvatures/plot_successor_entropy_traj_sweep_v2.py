"""Plot v2 Successor Entropy sweep on τ-Colosseum.

For each (n_trajectories, τ): compute Spearman/Pearson of per-manifold
mean laziness against ks_true across 20 manifolds, then pick the best τ per
n_trajectories. Output lines showing best-over-τ correlation vs n_trajectories,
plus a heatmap of (τ, n_trajectories) → correlation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


def _corr_per_condition(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Compute per-(n_traj, τ) correlation of mean laziness vs ks_true across manifolds."""
    rows = []
    fn = scipy.stats.spearmanr if method == "spearman" else scipy.stats.pearsonr
    for (n_traj, temperature), g in df.groupby(["n_trajectories", "temperature"]):
        a = g["ks_hat_mean"].to_numpy(dtype=float)
        b = g["ks_true"].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
            r = float("nan")
        else:
            r = fn(a[m], b[m])[0]
            r = float(r) if np.isfinite(r) else float("nan")
        rows.append(dict(
            n_trajectories=int(n_traj),
            temperature=float(temperature),
            corr=r,
            n=int(m.sum()),
        ))
    return pd.DataFrame(rows).sort_values(["n_trajectories", "temperature"])


def plot_best(corr_df: pd.DataFrame, out_stem: Path, ylabel: str, title: str,
              target: float | None = 0.5) -> None:
    best = corr_df.loc[corr_df.groupby("n_trajectories")["corr"].idxmax()].copy()

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(best["n_trajectories"], best["corr"],
            marker="o", linewidth=2, color="C0", label="best-τ")
    for _, r in best.iterrows():
        ax.annotate(f"τ={r.temperature:g}",
                    xy=(r.n_trajectories, r["corr"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    if target is not None:
        ax.axhline(target, color="green", ls=":", lw=1.2, alpha=0.7,
                   label=f"target ({target})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of trajectories")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def plot_heatmap(corr_df: pd.DataFrame, out_stem: Path, title: str) -> None:
    grid = corr_df.pivot(index="temperature", columns="n_trajectories", values="corr")

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        grid, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("n_trajectories")
    ax.set_ylabel("softmax τ")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics",
                   default="processed_data/successor_entropy_traj_sweep_v2.csv")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    out_dir = Path(args.out_dir)

    corr_sp = _corr_per_condition(df, "spearman")
    corr_pe = _corr_per_condition(df, "pearson")

    plot_best(
        corr_sp, out_dir / "successor_entropy_traj_sweep_v2_spearman_best",
        ylabel="Spearman(⟨lazi⟩, ks_true) across 20 manifolds",
        title="v2 Successor Entropy on τ-Colosseum (γ=0.9, best softmax τ)\n"
              "n_samples=2000, z_dim=16, 600 epochs, cosine LR",
    )
    plot_best(
        corr_pe, out_dir / "successor_entropy_traj_sweep_v2_pearson_best",
        ylabel="Pearson(⟨lazi⟩, ks_true) across 20 manifolds",
        title="v2 Successor Entropy on τ-Colosseum (γ=0.9, best softmax τ)",
    )
    plot_heatmap(
        corr_sp, out_dir / "successor_entropy_traj_sweep_v2_spearman_heatmap",
        title="v2 Successor Entropy on τ-Colosseum: Spearman(⟨lazi⟩, ks_true)",
    )


if __name__ == "__main__":
    main()
