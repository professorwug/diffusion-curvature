"""Plots for `benchmark_all_on_iid.py` results.

Emits:
- `figures/colosseum_spearman_heatmap.{svg,png}` — facet grid of
  (d, c, method) Spearman heatmaps, one panel per noise level.
- `figures/colosseum_sign_acc_heatmap.{svg,png}` — same for sign accuracy.
- `figures/sadspheres_methods.{svg,png}` — per-dim bar panels: AUC
  (sphere vs saddle) and per-instance sign accuracy.
- `processed_data/colosseum_summary.csv`, `sadspheres_summary.csv`.
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


METHOD_ORDER = [
    "dc2",
    "dc_wasserstein_ollivier",
    "dc_entropic_subtraction",
    "hickock",
    "frc_weighted_adaptive",
    "frc_unweighted_knn",
    "orc",
    "spc_hop",
    "spc_diffusion_t5",
    "laziness_knn_t5",
    "laziness_adaptive_t5",
    "successor_entropy",
    "successor_orc",
]

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


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.spearmanr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def _sign_acc(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(a[m]) == np.sign(b[m])))


def _auc(a: np.ndarray, labels: np.ndarray) -> float:
    m = np.isfinite(a)
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels[m].astype(int), a[m]))
    except ValueError:
        return float("nan")


def summarize_colosseum(df: pd.DataFrame) -> pd.DataFrame:
    cc = df[df.dataset == "colosseum"].copy()
    rows = []
    for (method, d, c, noise), g in cc.groupby(["method", "dim", "codim", "noise"]):
        a = g["ks_hat"].to_numpy(dtype=float)
        b = g["ks_true"].to_numpy(dtype=float)
        rows.append(dict(
            method=method, dim=d, codim=c, noise=noise,
            spearman=_spearman(a, b),
            sign_acc=_sign_acc(a, b),
            n=len(g),
        ))
    return pd.DataFrame(rows)


def summarize_sadspheres(df: pd.DataFrame) -> pd.DataFrame:
    ss = df[df.dataset == "sadspheres"].copy()
    rows = []
    for (method, dim), g in ss.groupby(["method", "dim"]):
        a = g["ks_hat"].to_numpy(dtype=float)
        b = g["ks_true"].to_numpy(dtype=float)
        # label: 1 if ks_true > 0 (sphere), 0 if < 0 (saddle), NaN if 0 (plane) — excluded
        labels = np.where(b > 0, 1, np.where(b < 0, 0, -1))
        mask = labels >= 0
        rows.append(dict(
            method=method, dim=dim,
            sign_acc=_sign_acc(a, b),
            auc=_auc(a[mask], labels[mask]),
            n=len(g),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    present = list(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present] + [m for m in present if m not in METHOD_ORDER]


def plot_colosseum_heatmap(
    summary: pd.DataFrame, metric: str, out_stem: Path, title: str, cmap: str, vmin: float, vmax: float,
) -> None:
    methods = _ordered_methods(summary)
    noises = sorted(summary["noise"].unique())
    dims = sorted(summary["dim"].unique())
    codims = sorted(summary["codim"].unique())

    sns.set_context("talk")
    fig, axes = plt.subplots(
        len(methods), len(noises),
        figsize=(3.3 * len(noises), 1.6 * len(methods)),
        squeeze=False,
    )
    for m_i, method in enumerate(methods):
        for n_i, noise in enumerate(noises):
            ax = axes[m_i, n_i]
            sub = summary[(summary.method == method) & (summary.noise == noise)]
            grid = sub.pivot(index="dim", columns="codim", values=metric).reindex(
                index=dims, columns=codims,
            )
            sns.heatmap(
                grid, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", cbar=(n_i == len(noises) - 1),
                linewidths=0.5, annot_kws={"size": 8},
            )
            if m_i == 0:
                ax.set_title(f"noise={noise}", fontsize=11)
            else:
                ax.set_title("")
            if n_i == 0:
                ax.set_ylabel(METHOD_LABEL.get(method, method), fontsize=10)
            else:
                ax.set_ylabel("")
            if m_i == len(methods) - 1:
                ax.set_xlabel("codim")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    fig.suptitle(title, y=1.002, fontsize=14)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def plot_sadspheres(summary: pd.DataFrame, out_stem: Path) -> None:
    methods = _ordered_methods(summary)
    dims = sorted(summary["dim"].unique())

    sns.set_context("talk")
    fig, axes = plt.subplots(
        1, 2, figsize=(8 + 2.2 * len(dims), 6 + 0.3 * len(methods)),
        squeeze=False,
    )
    ax_auc, ax_sign = axes[0]

    # AUC
    auc_grid = summary.pivot(index="method", columns="dim", values="auc").reindex(
        index=methods, columns=dims,
    )
    auc_grid.index = [METHOD_LABEL.get(m, m) for m in methods]
    sns.heatmap(
        auc_grid, ax=ax_auc, cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5,
    )
    ax_auc.set_title("SadSpheres AUC (sphere vs saddle)")
    ax_auc.set_xlabel("dim")
    ax_auc.set_ylabel("")

    # sign_acc
    sign_grid = summary.pivot(index="method", columns="dim", values="sign_acc").reindex(
        index=methods, columns=dims,
    )
    sign_grid.index = [METHOD_LABEL.get(m, m) for m in methods]
    sns.heatmap(
        sign_grid, ax=ax_sign, cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5,
    )
    ax_sign.set_title("SadSpheres per-instance sign accuracy")
    ax_sign.set_xlabel("dim")
    ax_sign.set_ylabel("")

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/iid_metrics.csv")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--summary-dir", default="processed_data")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    out_dir = Path(args.out_dir)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    cc_summary = summarize_colosseum(df)
    ss_summary = summarize_sadspheres(df)
    cc_summary.to_csv(summary_dir / "colosseum_summary.csv", index=False)
    ss_summary.to_csv(summary_dir / "sadspheres_summary.csv", index=False)

    if not cc_summary.empty:
        plot_colosseum_heatmap(
            cc_summary, "spearman",
            out_stem=out_dir / "colosseum_spearman_heatmap",
            title="Colosseum: Spearman(ks_hat, ks_true) per (d, codim, noise)",
            cmap="RdBu_r", vmin=-1, vmax=1,
        )
        plot_colosseum_heatmap(
            cc_summary, "sign_acc",
            out_stem=out_dir / "colosseum_sign_acc_heatmap",
            title="Colosseum: sign accuracy per (d, codim, noise)",
            cmap="RdBu_r", vmin=0, vmax=1,
        )

    if not ss_summary.empty:
        plot_sadspheres(ss_summary, out_dir / "sadspheres_methods")


if __name__ == "__main__":
    main()
