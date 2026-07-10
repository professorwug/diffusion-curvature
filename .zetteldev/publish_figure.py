#!/usr/bin/env python3
"""Publish figures to Cloudflare R2 for stable, cross-device access.

Uploads figures to R2 with provenance metadata (git commit, source script,
experiment name). Returns a public URL that can be embedded in Obsidian notes.

Usage:
    # Single figure
    uv run python .zetteldev/publish_figure.py experiments/20-foo/figures/fig_bar.png

    # Sync all changed figures across all experiments
    uv run python .zetteldev/publish_figure.py --sync

    # Sync one experiment
    uv run python .zetteldev/publish_figure.py --sync experiments/20-foo

The URL structure is:
    https://blots.kincaid.ink/{repo_name}/{experiment}/{filename}

A figure registry (.zetteldev/figure_registry.json) tracks file hashes so only
changed figures are re-uploaded.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

BUCKET_NAME = "zetteldev-figures"
PUBLIC_BASE = "https://blots.kincaid.ink"
IMAGE_EXTENSIONS = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(result.stdout.strip())


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def get_repo_name() -> str:
    return get_git_root().name


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def infer_experiment(figure_path: Path, git_root: Path) -> str | None:
    """Infer experiment name from path like experiments/20-foo/figures/fig.png."""
    try:
        rel = figure_path.resolve().relative_to(git_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "experiments":
        return parts[1]
    return None


def infer_source_script(figure_path: Path) -> str | None:
    """Guess the source script from the figure filename."""
    stem = figure_path.stem
    experiment_dir = figure_path.parent.parent
    candidates = [
        experiment_dir / "scripts" / f"{stem}.py",
        experiment_dir / f"{stem}.py",
    ]
    for c in candidates:
        if c.exists():
            return c.name
    return None


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------

def get_r2_client():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")

    if not all([account_id, access_key, secret_key]):
        print("Error: Missing R2 credentials in .env file.", file=sys.stderr)
        print("Required: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_R2_ACCESS_KEY_ID, CLOUDFLARE_R2_SECRET_ACCESS_KEY", file=sys.stderr)
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def ensure_bucket_exists(client):
    try:
        client.head_bucket(Bucket=BUCKET_NAME)
    except client.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            print(f"Creating bucket '{BUCKET_NAME}'...")
            client.create_bucket(Bucket=BUCKET_NAME)
        else:
            raise


def get_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def registry_path() -> Path:
    return get_git_root() / ".zetteldev" / "figure_registry.json"


def load_registry() -> dict:
    p = registry_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_registry(registry: dict):
    p = registry_path()
    p.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def make_key(figure_path: Path, repo_name: str, experiment: str | None) -> str:
    if experiment:
        return f"{repo_name}/{experiment}/{figure_path.name}"
    return f"{repo_name}/{figure_path.name}"


def publish_figure(
    figure_path: Path,
    client,
    source_script: str | None = None,
    dry_run: bool = False,
) -> str:
    """Upload a figure to R2 and return the public URL."""
    git_root = get_git_root()
    repo_name = get_repo_name()
    commit = get_git_commit()
    experiment = infer_experiment(figure_path, git_root)

    if source_script is None:
        source_script = infer_source_script(figure_path) or "unknown"

    key = make_key(figure_path, repo_name, experiment)

    metadata = {
        "git-commit": commit,
        "source-script": source_script,
        "repo": repo_name,
        "original-path": str(figure_path),
    }
    if experiment:
        metadata["experiment"] = experiment

    if dry_run:
        print(f"  [dry-run] {figure_path} -> {key}")
        return f"{PUBLIC_BASE}/{key}"

    client.upload_file(
        str(figure_path),
        BUCKET_NAME,
        key,
        ExtraArgs={
            "ContentType": get_content_type(figure_path),
            "CacheControl": "no-cache, must-revalidate",
            "Metadata": metadata,
        },
    )

    public_url = f"{PUBLIC_BASE}/{key}"
    print(f"  Uploaded: {figure_path.name}  ({commit}, {source_script})")
    return public_url


# ---------------------------------------------------------------------------
# Sync: scan experiments, publish changed figures
# ---------------------------------------------------------------------------

def discover_figures(root: Path, experiment_filter: str | None = None) -> list[Path]:
    """Find all image files in experiments/*/figures/ directories."""
    experiments_dir = root / "experiments"
    if not experiments_dir.exists():
        return []

    figures = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if experiment_filter and exp_dir.name != experiment_filter:
            continue
        fig_dir = exp_dir / "figures"
        if not fig_dir.exists():
            continue
        for f in sorted(fig_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                figures.append(f)
    return figures


def sync_figures(
    experiment_filter: str | None = None,
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Sync all changed figures. Returns list of (path, url) for uploaded files."""
    git_root = get_git_root()
    registry = load_registry()
    figures = discover_figures(git_root, experiment_filter)

    if not figures:
        scope = experiment_filter or "all experiments"
        print(f"No figures found in {scope}.")
        return []

    # Determine which figures changed
    changed = []
    for fig in figures:
        rel = str(fig.relative_to(git_root))
        current_hash = file_hash(fig)
        if registry.get(rel, {}).get("hash") != current_hash:
            changed.append((fig, rel, current_hash))

    if not changed:
        print(f"All {len(figures)} figures up to date.")
        return []

    print(f"Publishing {len(changed)}/{len(figures)} changed figures...")

    client = None
    if not dry_run:
        client = get_r2_client()
        ensure_bucket_exists(client)

    uploaded = []
    for fig, rel, current_hash in changed:
        url = publish_figure(fig, client, dry_run=dry_run)
        if not dry_run:
            registry[rel] = {
                "hash": current_hash,
                "url": url,
                "commit": get_git_commit(),
            }
        uploaded.append((rel, url))

    if not dry_run:
        save_registry(registry)
        print(f"\nRegistry updated ({len(uploaded)} entries).")

    return uploaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Publish figures to Cloudflare R2")
    parser.add_argument("path", nargs="?", type=Path, help="Figure file or experiment directory")
    parser.add_argument("--source", type=str, default=None, help="Source script name")
    parser.add_argument("--sync", action="store_true", help="Sync all changed figures")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument("--obsidian", action="store_true", help="Output Obsidian-ready markdown")
    args = parser.parse_args()

    load_dotenv(get_git_root() / ".env")

    if args.sync:
        # --sync with optional experiment filter
        experiment_filter = None
        if args.path:
            # Accept either "experiments/20-foo" or "20-foo"
            name = args.path.name
            if args.path.parent.name == "experiments":
                name = args.path.name
            experiment_filter = name
        uploaded = sync_figures(experiment_filter, dry_run=args.dry_run)
        if args.obsidian:
            for rel, url in uploaded:
                name = url.rsplit("/", 1)[-1]
                print(f"![{name}]({url})")
        return

    if args.path is None:
        parser.error("Provide a figure path, or use --sync")

    if args.path.is_dir():
        # Publish all figures in an experiment directory
        experiment_filter = args.path.name
        if args.path.parent.name == "experiments":
            experiment_filter = args.path.name
        sync_figures(experiment_filter, dry_run=args.dry_run)
    else:
        if not args.path.exists():
            print(f"Error: {args.path} does not exist", file=sys.stderr)
            sys.exit(1)
        client = None
        if not args.dry_run:
            client = get_r2_client()
            ensure_bucket_exists(client)
        url = publish_figure(args.path, client, source_script=args.source, dry_run=args.dry_run)

        # Update registry for single-file uploads too
        if not args.dry_run:
            git_root = get_git_root()
            rel = str(args.path.relative_to(git_root))
            registry = load_registry()
            registry[rel] = {
                "hash": file_hash(args.path),
                "url": url,
                "commit": get_git_commit(),
            }
            save_registry(registry)

        print(f"  URL: {url}")
        if args.obsidian:
            print(f"\n![{args.path.name}]({url})")


if __name__ == "__main__":
    main()
