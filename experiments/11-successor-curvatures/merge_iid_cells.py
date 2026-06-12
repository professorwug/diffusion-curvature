"""Concatenate per-cell IID Colosseum CSVs into a single metrics file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()

    frames = []
    for path in args.inputs:
        f = Path(path)
        if f.exists() and f.stat().st_size > 0:
            frames.append(pd.read_csv(f))
        else:
            print(f"  skipping missing/empty: {f}")
    if not frames:
        raise SystemExit("No input shards.")
    df = pd.concat(frames, ignore_index=True)
    # Per-cell `instance` indices reset to 0..49, so we cannot dedup on
    # (dataset, instance, method) alone — that would collapse all 20 cells
    # to whichever shard sorted last. Use the cell coordinates too.
    df = df.drop_duplicates(
        subset=["dataset", "dim", "noise", "codim", "instance", "method"],
        keep="last",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Merged {len(frames)} cells → {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
