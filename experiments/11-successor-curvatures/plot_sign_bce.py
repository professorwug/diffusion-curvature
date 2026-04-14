"""Plot sign-classification BCE vs n_samples on τ-Saddle-Sphere.

One panel per n_trajectories, lines per method. Lower BCE is better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_LABEL = {
    "laziness": "Diffusion Laziness",
    "successor_entropy": "Successor Entropy",
}
METHOD_COLOR = {
    "laziness": "#1f77b4",
    "successor_entropy": "#d62728",
}


def plot(df: pd.DataFrame, out_stem: Path) -> None:
    sub = df[df["dataset"] == "sadspheres"].copy()
    if sub.empty:
        print("No SadSpheres rows; skipping.")
        return

    agg = (
        sub.groupby(["method", "n_samples", "n_trajectories"])
        .agg(bce_mean=("bce", "mean"), bce_sem=("bce", "sem"),
             acc=("pred", lambda p: float(
                 np.mean((p.values > 0.5).astype(float) == sub.loc[p.index, "label"].values)
             )))
        .reset_index()
    )
    n_trajs = sorted(agg["n_trajectories"].unique())

    sns.set_context("talk")
    fig, axes = plt.subplots(
        1, len(n_trajs), figsize=(5.5 * len(n_trajs), 5.0), sharey=True,
        squeeze=False,
    )
    for ax, n_t in zip(axes[0], n_trajs):
        a = agg[agg["n_trajectories"] == n_t]
        for m in a["method"].unique():
            rows = a[a["method"] == m].sort_values("n_samples")
            ax.errorbar(
                rows["n_samples"], rows["bce_mean"],
                yerr=rows["bce_sem"].fillna(0.0),
                marker="o", capsize=3, linewidth=2,
                label=METHOD_LABEL.get(m, m),
                color=METHOD_COLOR.get(m, None),
            )
        ax.axhline(np.log(2), color="gray", linewidth=0.5, linestyle="--",
                   alpha=0.5, label=None)
        ax.set_xscale("log")
        ax.set_xlabel("Corpus size (n_samples)")
        ax.set_title(f"{n_t} trajectories")
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel("Binary cross-entropy (sign classification)")
    axes[0, 0].legend(loc="upper right", fontsize=11)
    fig.suptitle("τ-Saddle-Sphere: sign classification vs corpus size", y=1.02)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/metrics.csv")
    p.add_argument("--out", default="figures/tau_sadspheres_bce")
    args = p.parse_args()
    df = pd.read_csv(args.metrics)
    plot(df, Path(args.out))


if __name__ == "__main__":
    main()
