"""Plot config-level sign-classification AUC vs n_samples on τ-Saddle-Sphere.

For each (method, n_samples, n_trajectories), computes ROC AUC of
per-instance mean estimated curvature predicting the sphere/saddle label.
Higher is better; 0.5 = chance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


METHOD_LABEL = {
    "laziness": "Diffusion Laziness",
    "successor_entropy": "Successor Entropy",
    "successor_orc": "Successor ORC (B)",
    "laziness_knn_t3": "Laziness (kNN, t=3)",
    "laziness_knn_t5": "Laziness (kNN, t=5)",
    "laziness_knn_t10": "Laziness (kNN, t=10)",
    "laziness_adaptive_t5": "Laziness (adaptive, t=5)",
}
METHOD_COLOR = {
    "successor_entropy": "#d62728",
    "successor_orc": "#2ca02c",
    "laziness_knn_t3": "#9ecae1",
    "laziness_knn_t5": "#4292c6",
    "laziness_knn_t10": "#08519c",
    "laziness_adaptive_t5": "#fd8d3c",
}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    m = np.isfinite(scores) & np.isfinite(labels)
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels[m].astype(int), scores[m]))
    except ValueError:
        return float("nan")


def _bootstrap_auc(
    scores: np.ndarray, labels: np.ndarray, n_boot: int = 200, seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n < 2:
        return float("nan"), float("nan")
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = _auc(scores[idx], labels[idx])
    return float(np.nanmean(boot)), float(np.nanstd(boot))


def plot(df: pd.DataFrame, out_stem: Path) -> None:
    sub = df[df["dataset"] == "sadspheres"].copy()
    if sub.empty:
        print("No SadSpheres rows; skipping.")
        return

    rows = []
    for (method, n_s, n_t), g in sub.groupby(["method", "n_samples", "n_trajectories"]):
        mean_a, std_a = _bootstrap_auc(
            g["ks_hat_mean"].values, g["label"].values, seed=int(n_s + n_t),
        )
        rows.append(dict(method=method, n_samples=n_s, n_trajectories=n_t,
                         auc=mean_a, auc_std=std_a, n_instances=len(g)))
    agg = pd.DataFrame(rows)
    n_trajs = sorted(agg["n_trajectories"].unique())

    sns.set_context("talk")
    fig, axes = plt.subplots(
        1, len(n_trajs), figsize=(5.5 * len(n_trajs), 5.0), sharey=True,
        squeeze=False,
    )
    for ax, n_t in zip(axes[0], n_trajs):
        a = agg[agg["n_trajectories"] == n_t]
        for m in a["method"].unique():
            rows_m = a[a["method"] == m].sort_values("n_samples")
            ax.errorbar(
                rows_m["n_samples"], rows_m["auc"],
                yerr=rows_m["auc_std"].fillna(0.0),
                marker="o", capsize=3, linewidth=2,
                label=METHOD_LABEL.get(m, m),
                color=METHOD_COLOR.get(m, None),
            )
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Corpus size (n_samples)")
        ax.set_title(f"{n_t} trajectories")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.0, 1.05)

    axes[0, 0].set_ylabel("AUC: ⟨ks_hat⟩ predicts sphere vs saddle")
    axes[0, 0].legend(loc="lower right", fontsize=11)
    fig.suptitle("τ-Saddle-Sphere: sign classification (bootstrap mean ± std)", y=1.02)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/metrics.csv")
    p.add_argument("--out", default="figures/tau_sadspheres_auc")
    args = p.parse_args()
    df = pd.read_csv(args.metrics)
    plot(df, Path(args.out))


if __name__ == "__main__":
    main()
