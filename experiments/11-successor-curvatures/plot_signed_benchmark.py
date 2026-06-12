"""Figures for the natively-signed curvature benchmark (signed_metrics.csv).

Outputs (figures/signed/):
  1. signed_colosseum_pearson.{svg,png}   — method × (dim, noise) Pearson heatmap
  2. signed_colosseum_signacc.{svg,png}   — method × (dim, noise) sign-accuracy heatmap
  3. signed_sadspheres.{svg,png}          — AUC + sign accuracy by method × dim
  4. signed_scatter.{svg,png}             — ks_hat vs ks_true scatter for the new
                                            method at d=2, colored by noise
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
    "signed_orc_t8": "Diffusion ORC (t=8)",
    "signed_orc_phys_t8": "Diffusion ORC phys (t=8)",
    "signed_orc_t16": "Diffusion ORC (t=16)",
    "signed_orc_phys_t16": "Diffusion ORC phys (t=16)",
    "dc_wasserstein_ollivier": "DC (W. + Ollivier)",
    "orc": "ORC",
    "laziness_knn_t5": "Laziness (kNN, t=5)",
    "hickock": "Hickock",
}


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(scipy.stats.pearsonr(a[m], b[m])[0])


def _sign_acc(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.mean(np.sign(a[m]) == np.sign(b[m])))


def _heat(grid, title, cbar_label, out_stem, vmin=-1, vmax=1, cmap="RdBu_r",
          center=0):
    sns.set_context("talk")
    fig, ax = plt.subplots(
        figsize=(0.85 * len(grid.columns) + 4, 0.55 * len(grid) + 2))
    sns.heatmap(grid, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={"label": cbar_label}, annot_kws={"size": 10})
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stem}.{{svg,png}}")


def colosseum_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    cc = df[df.dataset == "colosseum"]
    rows = []
    for (method, d, noise), g in cc.groupby(["method", "dim", "noise"]):
        a, b = g.ks_hat.to_numpy(float), g.ks_true.to_numpy(float)
        rows.append(dict(method=method, dim=int(d), noise=float(noise),
                         pearson=_pearson(a, b), sign_acc=_sign_acc(a, b)))
    s = pd.DataFrame(rows)
    s["col"] = [f"d={d}\nε={n:g}" for d, n in zip(s.dim, s.noise)]
    col_order = (s[["col", "dim", "noise"]].drop_duplicates()
                 .sort_values(["dim", "noise"]).col.tolist())

    for metric, kw, fname, label in [
        ("pearson", dict(vmin=-1, vmax=1, cmap="RdBu_r", center=0),
         "signed_colosseum_pearson", "Pearson r"),
        ("sign_acc", dict(vmin=0, vmax=1, cmap="viridis", center=None),
         "signed_colosseum_signacc", "sign accuracy"),
    ]:
        grid = s.pivot(index="method", columns="col", values=metric)[col_order]
        order = grid.mean(axis=1).sort_values(ascending=False).index
        grid = grid.loc[order]
        grid.index = [METHOD_LABEL.get(m, m) for m in grid.index]
        _heat(grid,
              f"Curvature Colosseum — {label} vs ground-truth scalar curvature\n"
              f"(30 manifolds/cell, identical manifolds across methods)",
              label, outdir / fname, **kw)


def sadspheres_table(df: pd.DataFrame, outdir: Path) -> None:
    ss = df[(df.dataset == "sadspheres") & (df.ks_true != 0)]
    rows = []
    for (method, d), g in ss.groupby(["method", "dim"]):
        m = np.isfinite(g.ks_hat)
        g = g[m]
        if g.ks_true.nunique() < 2 or len(g) < 6:
            continue
        auc = roc_auc_score((g.ks_true > 0).astype(int), g.ks_hat)
        rows.append(dict(method=method, dim=int(d), auc=auc,
                         sign_acc=_sign_acc(g.ks_hat.to_numpy(float),
                                            g.ks_true.to_numpy(float))))
    s = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.6 * s.method.nunique() + 3))
    for ax, metric, label in [(axes[0], "auc", "AUC (sphere vs saddle)"),
                              (axes[1], "sign_acc", "sign accuracy")]:
        grid = s.pivot(index="method", columns="dim", values=metric)
        order = grid.mean(axis=1).sort_values(ascending=False).index
        grid = grid.loc[order]
        grid.index = [METHOD_LABEL.get(m, m) for m in grid.index]
        sns.heatmap(grid, ax=ax, cmap="viridis", vmin=0, vmax=1, annot=True,
                    fmt=".2f", linewidths=0.5, cbar=False,
                    annot_kws={"size": 11})
        ax.set_title(label)
        ax.set_xlabel("intrinsic dim")
        ax.set_ylabel("")
    fig.suptitle("SadSpheres — sphere/saddle separation and native sign", y=1.02)
    fig.tight_layout()
    for ext, kw in [("svg", {}), ("png", dict(dpi=200))]:
        fig.savefig(outdir / f"signed_sadspheres.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote {outdir}/signed_sadspheres.{{svg,png}}")


def scatter(df: pd.DataFrame, outdir: Path, method: str = "signed_orc_phys_t8") -> None:
    cc = df[(df.dataset == "colosseum") & (df.method == method) & (df.dim == 2)]
    if not len(cc):
        return
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8, 7))
    pal = sns.color_palette("flare", cc.noise.nunique())
    for c, (noise, g) in zip(pal, cc.groupby("noise")):
        ax.scatter(g.ks_true, g.ks_hat, color=c, alpha=0.75, s=45,
                   label=f"ε={noise:g}")
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("true scalar curvature at origin")
    ax.set_ylabel(METHOD_LABEL.get(method, method))
    r = _pearson(cc.ks_hat.to_numpy(float), cc.ks_true.to_numpy(float))
    sa = _sign_acc(cc.ks_hat.to_numpy(float), cc.ks_true.to_numpy(float))
    ax.set_title(f"d=2 Colosseum — native sign without a comparison space\n"
                 f"Pearson r = {r:.2f}, sign accuracy = {sa:.2f}")
    ax.legend(title="ambient noise", fontsize=11)
    fig.tight_layout()
    for ext, kw in [("svg", {}), ("png", dict(dpi=200))]:
        fig.savefig(outdir / f"signed_scatter.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote {outdir}/signed_scatter.{{svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/signed_metrics.csv")
    p.add_argument("--outdir", default="figures/signed")
    args = p.parse_args()
    df = pd.read_csv(args.metrics)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    colosseum_heatmaps(df, outdir)
    sadspheres_table(df, outdir)
    scatter(df, outdir)


if __name__ == "__main__":
    main()
