"""Heatmaps for the trajectory-regime colosseum run (kmetric_colosseum.csv):
Pearson and balanced sign per (method x n_traj) x dim, with the full-data iid
references (signed_orc_t8 / _auto on the same manifolds) as bottom rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

LABEL = {
    "kmetric_dorc": "Kernel-metric DORC",
    "fbkernel_orc": "FB-Kernel ORC (2-net)",
    "traj_dorc": "Diffusion ORC (visited cloud)",
    "signed_orc_t8": "iid ref: Diffusion ORC (t=8, full data)",
    "signed_orc_auto": "iid ref: Diffusion ORC (auto, full data)",
}


def _metrics(g):
    g = g[np.isfinite(g.ks_hat)]
    r = pearsonr(g.ks_hat, g.ks_true)[0] if len(g) > 3 and g.ks_hat.std() > 0 else np.nan
    pos, neg = g[g.ks_true > 0], g[g.ks_true < 0]
    bal = (0.5 * ((pos.ks_hat > 0).mean() + (neg.ks_hat < 0).mean())
           if len(pos) and len(neg) else np.nan)
    return r, bal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="processed_data/kmetric_colosseum.csv")
    ap.add_argument("--refs", default="processed_data/signed_metrics.csv")
    ap.add_argument("--out", default="figures/signed/kmetric_colosseum")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df["ks_hat"] = pd.to_numeric(df.ks_hat, errors="coerce")
    ref = pd.read_csv(args.refs)
    ref = ref[(ref.dataset == "colosseum") & (ref.m < 15)
              & ref.method.isin(["signed_orc_t8", "signed_orc_auto"])].copy()
    ref["ks_hat"] = pd.to_numeric(ref.ks_hat, errors="coerce")

    rows = []
    for (m, nt, d), g in df.groupby(["method", "n_traj", "dim"]):
        r, bal = _metrics(g)
        rows.append(dict(row=f"{LABEL[m]} (n_traj={nt})", order=(0, m, nt),
                         dim=d, pearson=r, balsign=bal))
    for (m, d), g in ref.groupby(["method", "dim"]):
        r, bal = _metrics(g)
        rows.append(dict(row=LABEL[m], order=(1, m, 0), dim=d,
                         pearson=r, balsign=bal))
    s = pd.DataFrame(rows)
    row_order = (s[["row", "order"]].drop_duplicates()
                 .sort_values("order").row.tolist())

    sns.set_context("talk")
    fig, axes = plt.subplots(1, 2, figsize=(19, 0.55 * len(row_order) + 3))
    for ax, metric, label, kw in [
        (axes[0], "pearson", "Pearson r",
         dict(cmap="RdBu_r", center=0, vmin=-1, vmax=1)),
        (axes[1], "balsign", "balanced sign accuracy",
         dict(cmap="RdBu_r", center=0.5, vmin=0, vmax=1)),
    ]:
        grid = s.pivot_table(index="row", columns="dim", values=metric)
        grid = grid.loc[row_order]
        sns.heatmap(grid, ax=ax, annot=True, fmt=".2f", linewidths=0.5,
                    cbar=False, annot_kws={"size": 11}, **kw)
        ax.set_title(label)
        ax.set_xlabel("intrinsic dim")
        ax.set_ylabel("")
    axes[1].set_yticklabels([])
    fig.suptitle("Trajectory-regime Colosseum: methods see only visited nodes "
                 "(coverage 72% at n_traj=100, 99.7% at 500); 15 manifolds/cell,"
                 " noise pooled", fontsize=13, y=1.02)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw2 in [("svg", {}), ("png", dict(dpi=200))]:
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", **kw2)
    print(f"wrote {out}.{{svg,png}}")


if __name__ == "__main__":
    main()
