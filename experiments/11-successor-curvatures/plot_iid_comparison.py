"""Bar charts comparing methods on iid SadSpheres (AUC + sign accuracy)
and Colosseum (Spearman).

Reads `processed_data/iid_metrics.csv` (per-instance rows) and computes
dataset-level summary metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.metrics import roc_auc_score


METHOD_LABEL = {
    "laziness_knn_t3": "Laziness (kNN, t=3)",
    "laziness_knn_t5": "Laziness (kNN, t=5)",
    "laziness_knn_t10": "Laziness (kNN, t=10)",
    "laziness_adaptive_t5": "Laziness (adaptive, t=5)",
    "successor_entropy": "Successor Entropy",
    "successor_orc": "Successor ORC (B)",
}
METHOD_COLOR = {
    "laziness_knn_t3": "#9ecae1",
    "laziness_knn_t5": "#4292c6",
    "laziness_knn_t10": "#08519c",
    "laziness_adaptive_t5": "#fd8d3c",
    "successor_entropy": "#d62728",
    "successor_orc": "#2ca02c",
}


def _bootstrap_ci(
    metric_fn, a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    if n < 3:
        return (float("nan"), float("nan"))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot[i] = metric_fn(a[idx], b[idx])
        except Exception:
            boot[i] = np.nan
    return float(np.nanmean(boot)), float(np.nanstd(boot))


def _auc(a, b) -> float:
    try:
        if len(np.unique(b)) < 2:
            return float("nan")
        return float(roc_auc_score(b.astype(int), a))
    except Exception:
        return float("nan")


def _spearman(a, b) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.spearmanr(a, b)
    return float(r) if np.isfinite(r) else float("nan")


def _sorted_methods(df: pd.DataFrame) -> list[str]:
    order = [m for m in METHOD_LABEL if m in df["method"].unique()]
    order += [m for m in df["method"].unique() if m not in order]
    return order


def plot(df: pd.DataFrame, out_stem: Path) -> None:
    sns.set_context("talk")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    ax_auc, ax_sign, ax_sp = axes

    # ---- SadSpheres panel 1: sphere-vs-saddle AUC ----
    ss = df[df.dataset == "sadspheres"].copy()
    methods = _sorted_methods(ss)
    auc_means, auc_stds = [], []
    for m in methods:
        g = ss[ss.method == m]
        mean_a, std_a = _bootstrap_ci(
            _auc, g["ks_hat_mean"].to_numpy(), g["label"].to_numpy(), seed=1,
        )
        auc_means.append(mean_a)
        auc_stds.append(std_a)
    colors = [METHOD_COLOR.get(m, "gray") for m in methods]
    labels = [METHOD_LABEL.get(m, m) for m in methods]
    ax_auc.barh(labels, auc_means, xerr=auc_stds, color=colors, edgecolor="black")
    ax_auc.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax_auc.set_xlim(0, 1.05)
    ax_auc.set_xlabel("AUC: mean(ks_hat) predicts sphere vs saddle")
    ax_auc.set_title("SadSpheres ranking")
    ax_auc.grid(axis="x", alpha=0.3)

    # ---- SadSpheres panel 2: per-point sign accuracy ----
    sign_means = [float(ss[ss.method == m]["sign_acc"].mean()) for m in methods]
    sign_sems = [float(ss[ss.method == m]["sign_acc"].sem()) for m in methods]
    ax_sign.barh(labels, sign_means, xerr=sign_sems, color=colors, edgecolor="black")
    ax_sign.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax_sign.set_xlim(0, 1.05)
    ax_sign.set_xlabel("Per-point sign accuracy")
    ax_sign.set_title("SadSpheres sign test")
    ax_sign.grid(axis="x", alpha=0.3)

    # ---- Colosseum: Spearman ----
    cc = df[df.dataset == "colosseum"].copy()
    methods_cc = _sorted_methods(cc)
    colors_cc = [METHOD_COLOR.get(m, "gray") for m in methods_cc]
    labels_cc = [METHOD_LABEL.get(m, m) for m in methods_cc]
    sp_means, sp_stds = [], []
    for m in methods_cc:
        g = cc[cc.method == m]
        mean_r, std_r = _bootstrap_ci(
            _spearman, g["ks_hat_mean"].to_numpy(), g["ks_true_scalar"].to_numpy(), seed=2,
        )
        sp_means.append(mean_r)
        sp_stds.append(std_r)
    ax_sp.barh(labels_cc, sp_means, xerr=sp_stds, color=colors_cc, edgecolor="black")
    ax_sp.axvline(0, color="gray", ls="--", lw=0.8)
    ax_sp.set_xlim(-1.05, 1.05)
    ax_sp.set_xlabel("Spearman(mean(ks_hat), ks_true)")
    ax_sp.set_title("Colosseum correlation")
    ax_sp.grid(axis="x", alpha=0.3)

    fig.suptitle("Curvature methods on iid-sampled benchmarks (bootstrap mean ± std)", y=1.02)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/iid_metrics.csv")
    p.add_argument("--out", default="figures/iid_benchmark_comparison")
    args = p.parse_args()
    df = pd.read_csv(args.metrics)
    plot(df, Path(args.out))


if __name__ == "__main__":
    main()
