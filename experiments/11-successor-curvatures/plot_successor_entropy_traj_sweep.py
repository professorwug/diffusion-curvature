"""Plot: Pearson correlation vs n_trajectories for Successor Entropy,
one line per softmax τ ∈ {0.2, 0.3, 0.5}.

A proper method: correlation should *grow* with n_trajectories and be
*consistent* across τ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    fn = scipy.stats.spearmanr if method == "spearman" else scipy.stats.pearsonr
    r = fn(a[m], b[m])[0] if method == "spearman" else fn(a[m], b[m])[0]
    return float(r) if np.isfinite(r) else float("nan")


def _bootstrap_corr(
    a: np.ndarray, b: np.ndarray, method: str, n_boot: int = 500, seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = _corr(a[idx], b[idx], method)
    return float(np.nanmean(boot)), float(np.nanstd(boot))


def plot(df: pd.DataFrame, out_stem: Path, method: str) -> None:
    rows = []
    for (tau, n_traj), g in df.groupby(["tau", "n_trajectories"]):
        mean_r, std_r = _bootstrap_corr(
            g["ks_hat_mean"].to_numpy(), g["ks_true"].to_numpy(),
            method=method, seed=int(n_traj * 100 + tau * 10),
        )
        rows.append(dict(
            tau=float(tau), n_trajectories=int(n_traj),
            corr=mean_r, corr_std=std_r, n=len(g),
        ))
    agg = pd.DataFrame(rows).sort_values(["tau", "n_trajectories"])

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    taus = sorted(agg["tau"].unique())
    palette = sns.color_palette("viridis", len(taus))
    for tau, color in zip(taus, palette):
        sub = agg[agg["tau"] == tau]
        ax.errorbar(
            sub["n_trajectories"], sub["corr"],
            yerr=sub["corr_std"].fillna(0.0),
            marker="o", capsize=3, linewidth=2,
            color=color, label=f"τ = {tau}",
        )
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of trajectories")
    label = "Pearson" if method == "pearson" else "Spearman"
    ax.set_ylabel(f"{label}(⟨ks_hat⟩, ks_true) across 20 manifolds")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        "Successor Entropy on τ-Colosseum (n_samples=2000, 20 manifolds)\n"
        f"{label} correlation vs trajectory count, per softmax τ",
        fontsize=13,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/successor_entropy_traj_sweep.csv")
    p.add_argument("--out", default="figures/successor_entropy_traj_sweep")
    p.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--tau", type=float, default=None,
                   help="If set, keep only rows with this softmax τ (single-line plot).")
    args = p.parse_args()
    df = pd.read_csv(args.metrics)
    if args.tau is not None:
        df = df[np.isclose(df["tau"], args.tau)]
        print(f"Filtered to τ={args.tau}: {len(df)} rows")
    plot(df, Path(args.out), method=args.method)


if __name__ == "__main__":
    main()
