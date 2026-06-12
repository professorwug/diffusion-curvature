"""Remove rows for a chosen set of methods from per-cell CSVs so they get
rerun on resume.

Used when the configuration of those methods changes (FB hparams, SPC dim
cap, etc.) but we want to keep the still-valid rows for other methods.

Examples:
    # Drop successor rows (default).
    python strip_successor_rows.py processed_data/iid_cells/*.csv

    # Drop SPC rows so a relaxed dim cap takes effect.
    python strip_successor_rows.py --methods spc_hop spc_diffusion_t5 \\
        processed_data/iid_cells/*.csv processed_data/tau_cells/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_METHODS = ["successor_entropy", "successor_orc"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   help=f"Methods to strip (default: {DEFAULT_METHODS}).")
    p.add_argument("paths", nargs="+",
                   help="CSV files (per-cell or merged) to strip in place.")
    args = p.parse_args()

    targets = set(args.methods)
    label = "/".join(args.methods)
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists() or path.stat().st_size == 0:
            print(f"  skip (missing/empty): {path}")
            continue
        df = pd.read_csv(path)
        before = len(df)
        df = df[~df["method"].isin(targets)]
        after = len(df)
        df.to_csv(path, index=False)
        print(f"  {path}: {before} → {after} rows  (-{before - after} {label})")


if __name__ == "__main__":
    main()
