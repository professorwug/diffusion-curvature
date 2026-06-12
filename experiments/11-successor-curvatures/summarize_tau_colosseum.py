"""Aggregate `tau_metrics.csv` to a per-(method, dim, noise) Pearson summary.

Mirrors `plot_colosseum_pearson_matrix.build_summary` so that the same
plotting script (`plot_colosseum_pearson_compact.py`) can consume the
trajectory-sampled summary by pointing `--summary` at this output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="processed_data/tau_metrics.csv")
    p.add_argument("--out", default="processed_data/v2/tau_colosseum_pearson_summary.csv")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    rows = []
    for (method, d, noise), g in df.groupby(["method", "dim", "noise"]):
        a = g["ks_hat"].to_numpy(dtype=float)
        b = g["ks_true"].to_numpy(dtype=float)
        rows.append(dict(
            method=method, dim=int(d), noise=float(noise),
            pearson=_pearson(a, b), n=len(g),
        ))
    summary = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(summary)} (method, dim, noise) cells)")


if __name__ == "__main__":
    main()
