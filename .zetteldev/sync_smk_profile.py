#!/usr/bin/env python3
"""Sync the default Snakemake workflow-profile symlink into every experiment.

Snakemake auto-discovers a workflow profile at ``profiles/default/`` relative to
the working directory. Experiments are run from inside their own dir
(``cd experiments/<name> && snakemake ...``), so each needs its own
``profiles/default`` pointing at the one canonical profile under
``.zetteldev/snakemake/default``.

This keeps a single real config file (DRY, version-controlled) while making the
profile auto-apply with zero flags from anywhere we actually run Snakemake.

Usage:
    python .zetteldev/sync_smk_profile.py            # backfill all experiments + repo root
    python .zetteldev/sync_smk_profile.py <dir>      # ensure the symlink in one dir
    python .zetteldev/sync_smk_profile.py --dry-run  # report only
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANON = REPO_ROOT / ".zetteldev" / "snakemake" / "default"


def ensure_symlink(workdir: Path, dry_run: bool = False) -> str:
    """Ensure ``workdir/profiles/default`` is a symlink to the canonical profile.

    Returns a one-word status: created | fixed | ok | skipped(real dir).
    """
    link = workdir / "profiles" / "default"
    # Relative target, resolved from the link's *parent* dir (profiles/).
    target = os.path.relpath(CANON, link.parent)

    if link.is_symlink():
        if os.readlink(link) == target:
            return "ok"
        if not dry_run:
            link.unlink()
            link.symlink_to(target)
        return "fixed"
    if link.exists():
        # A real directory/file already lives here — never clobber it.
        return "skipped(real path)"
    if not dry_run:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    return "created"


def iter_targets() -> list[Path]:
    """All dirs that should carry the profile: every experiment + repo root."""
    targets = [REPO_ROOT]
    exp_base = REPO_ROOT / "experiments"
    if exp_base.exists():
        for d in sorted(exp_base.iterdir()):
            if d.is_dir() and (d / "Snakefile").exists():
                targets.append(d)
    return targets


def main(argv: list[str]) -> int:
    if not CANON.is_dir():
        sys.exit(f"Canonical profile not found at {CANON}")

    dry_run = "--dry-run" in argv
    explicit = [a for a in argv if not a.startswith("-")]
    targets = [Path(p).resolve() for p in explicit] if explicit else iter_targets()

    for wd in targets:
        status = ensure_symlink(wd, dry_run=dry_run)
        rel = os.path.relpath(wd, REPO_ROOT)
        print(f"  {status:<18} {rel}/profiles/default")
    if dry_run:
        print("(dry run — no changes written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
