#!/usr/bin/env python3
"""
Hugging Face data synchronization for experiment processed_data folders.

This script manages versioned snapshots of experiment data on Hugging Face,
replacing Git LFS for large file storage.

Usage:
    python hf_data.py init                    - Initialize HF repo connection
    python hf_data.py status                  - Show sync status for all experiments
    python hf_data.py push <experiment>       - Push experiment to HF (versioned snapshot)
    python hf_data.py pull <path>             - Pull from HF (supports subdirs)
    python hf_data.py pull --all              - Pull all experiments
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
MANIFEST_FILENAME = ".hf_manifest.json"
HF_CONFIG_FILE = REPO_ROOT / ".hf"
ENV_VAR_NAME = "HF_DATA_REPO"  # Legacy fallback


def get_hf_repo() -> str:
    """Get the HF repo ID from .hf config file (preferred) or environment (fallback)."""
    # First, check .hf config file (git-tracked)
    if HF_CONFIG_FILE.exists():
        repo = HF_CONFIG_FILE.read_text().strip()
        if repo:
            return repo

    # Fallback to environment variable
    repo = os.getenv(ENV_VAR_NAME)
    if repo:
        return repo

    print("Error: HF data repo not configured.")
    print("Run 'just hugdata init' to configure.")
    sys.exit(1)


def get_experiments() -> list[str]:
    """Get list of experiment directories that have processed_data folders."""
    experiments = []
    if not EXPERIMENTS_DIR.exists():
        return experiments

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if exp_dir.is_dir():
            processed_data = exp_dir / "processed_data"
            if processed_data.exists() and any(processed_data.iterdir()):
                experiments.append(exp_dir.name)

    return experiments


def compute_dir_hash(directory: Path) -> str:
    """Compute a hash of directory contents for change detection."""
    hasher = hashlib.sha256()

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            hasher.update(str(file_path.relative_to(directory)).encode())
            hasher.update(str(file_path.stat().st_size).encode())
            hasher.update(str(int(file_path.stat().st_mtime)).encode())

    return hasher.hexdigest()[:16]


def get_local_manifest(experiment: str) -> Optional[dict]:
    """Get local manifest for an experiment if it exists."""
    manifest_path = EXPERIMENTS_DIR / experiment / "processed_data" / MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def save_local_manifest(experiment: str, manifest: dict) -> None:
    """Save local manifest for an experiment."""
    manifest_path = EXPERIMENTS_DIR / experiment / "processed_data" / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def get_dir_size(directory: Path) -> int:
    """Get total size of directory in bytes (excludes dotfiles for consistency with hash)."""
    total = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            total += file_path.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# Commands
# =============================================================================


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize the HF repo connection."""
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Run: uv add huggingface_hub")
        sys.exit(1)

    # Check if already configured
    existing_repo = None
    if HF_CONFIG_FILE.exists():
        existing_repo = HF_CONFIG_FILE.read_text().strip()
    if not existing_repo:
        existing_repo = os.getenv(ENV_VAR_NAME)

    if existing_repo:
        print(f"Current HF repo: {existing_repo}")
        response = input("Reconfigure? [y/N]: ").strip().lower()
        if response != "y":
            print("Keeping existing configuration.")
            return

    # Login to HF
    print("\n1. Logging into Hugging Face...")
    print("   (This will open a browser or prompt for a token)")
    login()

    # Get or create repo
    print("\n2. Configure dataset repository...")
    repo_id = input("   Enter HF repo ID (e.g., 'username/reason_reckon_data'): ").strip()

    if not repo_id or "/" not in repo_id:
        print("Error: Invalid repo ID. Format should be 'username/repo_name'")
        sys.exit(1)

    api = HfApi()

    # Check if repo exists, create if not
    try:
        api.repo_info(repo_id, repo_type="dataset")
        print(f"   Found existing dataset: {repo_id}")
    except Exception:
        print(f"   Creating new dataset: {repo_id}")
        try:
            api.create_repo(repo_id, repo_type="dataset", private=True)
            print(f"   Created private dataset: {repo_id}")
        except Exception as e:
            print(f"Error creating repo: {e}")
            sys.exit(1)

    # Save to .hf (git-tracked config file)
    HF_CONFIG_FILE.write_text(repo_id + "\n")

    print(f"\n3. Saved repo ID to .hf")
    print("   (This file is git-tracked so collaborators know where to find data)")
    print("\nSetup complete! You can now use:")
    print("  just hugdata status   - view sync status")
    print("  just hugdata push <experiment>   - push data")
    print("  just hugdata pull <path>         - pull data")


def cmd_status(args: argparse.Namespace) -> None:
    """Show sync status for all experiments."""
    try:
        from huggingface_hub import HfApi, list_repo_tree
    except ImportError:
        print("Error: huggingface_hub not installed. Run: uv add huggingface_hub")
        sys.exit(1)

    repo_id = get_hf_repo()
    api = HfApi()

    # Get remote experiments (top-level directories only)
    remote_experiments = set()
    try:
        for item in list_repo_tree(repo_id, repo_type="dataset"):
            # Only include top-level directories (not files like .gitattributes)
            if item.path and "/" not in item.path and not item.path.startswith("."):
                # Check if it's a directory by looking for the type attribute
                if hasattr(item, "type") and item.type == "directory":
                    remote_experiments.add(item.path)
                elif not hasattr(item, "type"):
                    # Fallback: assume directories don't have extensions
                    if "." not in item.path:
                        remote_experiments.add(item.path)
    except Exception as e:
        print(f"Warning: Could not fetch remote info: {e}")
        remote_experiments = set()

    # Get local experiments
    local_experiments = get_experiments()
    all_experiments = sorted(set(local_experiments) | remote_experiments)

    if not all_experiments:
        print("No experiments found locally or on HF.")
        return

    # Print header
    print(f"\nHF Repo: {repo_id}")
    print("=" * 70)
    print(f"{'Experiment':<40} {'Local':<10} {'Remote':<10} {'Status':<10}")
    print("-" * 70)

    for exp in all_experiments:
        local_size = "—"
        remote_size = "—"
        status = "unknown"

        # Check local
        local_path = EXPERIMENTS_DIR / exp / "processed_data"
        has_local = local_path.exists() and any(local_path.iterdir())
        if has_local:
            local_size = format_size(get_dir_size(local_path))
            local_hash = compute_dir_hash(local_path)

        # Check remote
        has_remote = exp in remote_experiments
        if has_remote:
            # Get remote manifest if exists
            try:
                from huggingface_hub import hf_hub_download
                manifest_path = hf_hub_download(
                    repo_id,
                    f"{exp}/{MANIFEST_FILENAME}",
                    repo_type="dataset",
                    local_dir=REPO_ROOT / ".hf_cache",
                    force_download=True,  # Always fetch latest to avoid stale cache
                )
                with open(manifest_path) as f:
                    remote_manifest = json.load(f)
                remote_size = format_size(remote_manifest.get("size_bytes", 0))
                remote_hash = remote_manifest.get("hash", "")
            except Exception:
                remote_size = "?"
                remote_hash = ""

        # Determine status
        if has_local and has_remote:
            local_manifest = get_local_manifest(exp)
            if local_manifest and local_manifest.get("hash") == local_hash:
                # Local matches what we last pushed
                if local_hash == remote_hash:
                    status = "synced"
                else:
                    status = "behind"
            else:
                status = "ahead"
        elif has_local:
            status = "local only"
        elif has_remote:
            status = "remote only"

        # Truncate experiment name if needed
        exp_display = exp[:38] + ".." if len(exp) > 40 else exp
        print(f"{exp_display:<40} {local_size:<10} {remote_size:<10} {status:<10}")

    print("-" * 70)


MAX_UPLOAD_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
BATCH_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB per batch


def _is_transient_error(error_msg: str) -> bool:
    return any(s in error_msg.lower() for s in (
        "panic", "joinerror", "timed out", "connection",
        "broken pipe", "reset by peer",
    ))


def _commit_batch_with_retry(
    api,
    repo_id: str,
    operations: list,
    commit_message: str,
    max_retries: int = MAX_UPLOAD_RETRIES,
) -> None:
    """Commit a batch of operations with retry + xet-disable fallback."""
    for attempt in range(1, max_retries + 1):
        saved_xet = None
        try:
            if attempt > 1:
                # Disable xet on retries — the panics are in xet-core
                saved_xet = os.environ.get("HF_HUB_DISABLE_XET")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
            return
        except Exception as e:
            error_msg = str(e)
            if not _is_transient_error(error_msg) or attempt == max_retries:
                raise
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"    Batch failed (attempt {attempt}): {error_msg[:120]}")
            print(f"    Retrying in {wait}s (xet disabled)...")
            time.sleep(wait)
        finally:
            if saved_xet is not None:
                os.environ["HF_HUB_DISABLE_XET"] = saved_xet
            else:
                os.environ.pop("HF_HUB_DISABLE_XET", None)


def upload_with_retry(
    api,
    folder_path: str,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> None:
    """Upload a folder to HF in size-limited batches with per-batch retry.

    Files are grouped into batches of ~100 MB. Each batch is committed
    separately with retry logic. On retries, xet is disabled to avoid
    xet-core panics. Use --no-xet flag to disable xet for all attempts.
    """
    from huggingface_hub import CommitOperationAdd

    folder = Path(folder_path)

    # Collect all files with sizes
    files = []
    for fp in sorted(folder.rglob("*")):
        if fp.is_file() and not fp.name.startswith("."):
            files.append((fp, fp.stat().st_size))

    if not files:
        print("  No files to upload.")
        return

    total_size = sum(s for _, s in files)

    # Group into batches under BATCH_SIZE_BYTES
    batches: list[list[tuple[Path, int]]] = []
    current_batch: list[tuple[Path, int]] = []
    current_size = 0
    for fp, size in files:
        # Files larger than the limit get their own batch
        if current_batch and current_size + size > BATCH_SIZE_BYTES:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append((fp, size))
        current_size += size
    if current_batch:
        batches.append(current_batch)

    if len(batches) == 1:
        print(f"  Uploading {len(files)} files ({format_size(total_size)}) in 1 batch...")
    else:
        print(f"  Uploading {len(files)} files ({format_size(total_size)}) "
              f"in {len(batches)} batches...")

    for batch_idx, batch in enumerate(batches, 1):
        batch_size = sum(s for _, s in batch)
        batch_files = len(batch)
        print(f"  Batch {batch_idx}/{len(batches)}: "
              f"{batch_files} files, {format_size(batch_size)}")

        operations = []
        for fp, _ in batch:
            rel = fp.relative_to(folder)
            repo_path = f"{path_in_repo}/{rel}" if path_in_repo else str(rel)
            operations.append(CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(fp),
            ))

        batch_msg = (f"{commit_message} (batch {batch_idx}/{len(batches)})"
                     if len(batches) > 1 else commit_message)
        _commit_batch_with_retry(api, repo_id, operations, batch_msg)
        print(f"    ✓ Batch {batch_idx} committed")


def cmd_push(args: argparse.Namespace) -> None:
    """Push an experiment's processed_data to HF."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed. Run: uv add huggingface_hub")
        sys.exit(1)

    experiment = args.experiment
    repo_id = get_hf_repo()

    # Validate experiment exists
    local_path = EXPERIMENTS_DIR / experiment / "processed_data"
    if not local_path.exists():
        print(f"Error: No processed_data folder found for '{experiment}'")
        print(f"Expected: {local_path}")
        sys.exit(1)

    if not any(local_path.iterdir()):
        print(f"Error: processed_data folder is empty for '{experiment}'")
        sys.exit(1)

    # Compute hash and size
    local_hash = compute_dir_hash(local_path)
    local_size = get_dir_size(local_path)

    print(f"Pushing: {experiment}")
    print(f"  Size: {format_size(local_size)}")
    print(f"  Hash: {local_hash}")
    print(f"  To: {repo_id}/{experiment}")

    # Create manifest
    manifest = {
        "experiment": experiment,
        "hash": local_hash,
        "size_bytes": local_size,
        "pushed_at": datetime.now().isoformat(),
        "files": []
    }

    # List files for manifest
    for file_path in sorted(local_path.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            rel_path = file_path.relative_to(local_path)
            manifest["files"].append({
                "path": str(rel_path),
                "size": file_path.stat().st_size
            })

    # Save manifest locally first
    save_local_manifest(experiment, manifest)

    # Upload to HF
    api = HfApi()
    commit_message = f"{experiment} @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    print(f"\nUploading to Hugging Face...")
    try:
        upload_with_retry(
            api,
            folder_path=str(local_path),
            repo_id=repo_id,
            path_in_repo=experiment,
            commit_message=commit_message,
        )
        print(f"\nSuccess! Commit: {commit_message}")
    except Exception as e:
        error_msg = str(e)
        if "timed out" in error_msg.lower():
            print(f"\nUpload timed out during finalization.")
            print("The data may have been uploaded. Check with: just hugdata status")
            print("If still showing 'local only', retry the push.")
        else:
            print(f"Error uploading: {e}")
        sys.exit(1)


def cmd_pull(args: argparse.Namespace) -> None:
    """Pull data from HF."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: uv add huggingface_hub")
        sys.exit(1)

    repo_id = get_hf_repo()

    if args.all:
        # Pull all experiments
        print(f"Pulling all experiments from {repo_id}...")
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=REPO_ROOT / ".hf_download_tmp",
            )
            # Move to proper locations
            tmp_dir = REPO_ROOT / ".hf_download_tmp"
            for exp_dir in tmp_dir.iterdir():
                if exp_dir.is_dir():
                    target = EXPERIMENTS_DIR / exp_dir.name / "processed_data"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        import shutil
                        shutil.rmtree(target)
                    exp_dir.rename(target)
                    print(f"  Pulled: {exp_dir.name}")
            # Cleanup
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir)
            print("Done!")
        except Exception as e:
            print(f"Error pulling: {e}")
            sys.exit(1)
    else:
        # Pull specific path (experiment or subdir)
        path = args.path
        if not path:
            print("Error: Please specify an experiment or path to pull.")
            print("Usage: just hugdata pull <experiment>")
            print("       just hugdata pull <experiment>/<subdir>")
            print("       just hugdata pull --all")
            sys.exit(1)

        # Parse path - could be "experiment" or "experiment/subdir"
        parts = path.split("/", 1)
        experiment = parts[0]
        subpath = parts[1] if len(parts) > 1 else ""

        target_base = EXPERIMENTS_DIR / experiment / "processed_data"

        print(f"Pulling: {path} from {repo_id}")

        try:
            if subpath:
                # Pull specific subdirectory
                allow_patterns = f"{experiment}/{subpath}/**"
                local_dir = REPO_ROOT / ".hf_download_tmp"
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    allow_patterns=[allow_patterns],
                )
                # Move to proper location
                source = local_dir / experiment / subpath
                target = target_base / subpath
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        import shutil
                        shutil.rmtree(target)
                    source.rename(target)
                    print(f"  Pulled to: {target}")
                # Cleanup
                import shutil
                shutil.rmtree(local_dir, ignore_errors=True)
            else:
                # Pull entire experiment
                allow_patterns = f"{experiment}/**"
                local_dir = REPO_ROOT / ".hf_download_tmp"
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    allow_patterns=[allow_patterns],
                )
                # Move to proper location
                source = local_dir / experiment
                if source.exists():
                    target_base.parent.mkdir(parents=True, exist_ok=True)
                    if target_base.exists():
                        import shutil
                        shutil.rmtree(target_base)
                    source.rename(target_base)
                    print(f"  Pulled to: {target_base}")
                # Cleanup
                import shutil
                shutil.rmtree(local_dir, ignore_errors=True)

            print("Done!")
        except Exception as e:
            print(f"Error pulling: {e}")
            sys.exit(1)


def cmd_pushall(args: argparse.Namespace) -> None:
    """Push all experiments that are local-only or ahead of remote."""
    try:
        from huggingface_hub import HfApi, hf_hub_download, list_repo_tree
    except ImportError:
        print("Error: huggingface_hub not installed. Run: uv add huggingface_hub")
        sys.exit(1)

    repo_id = get_hf_repo()

    # Get remote experiments
    remote_experiments = set()
    try:
        for item in list_repo_tree(repo_id, repo_type="dataset"):
            if item.path and "/" not in item.path and not item.path.startswith("."):
                if not hasattr(item, "type"):
                    if "." not in item.path:
                        remote_experiments.add(item.path)
    except Exception as e:
        print(f"Warning: Could not fetch remote info: {e}")

    # Find experiments that need pushing
    to_push = []
    local_experiments = get_experiments()

    for exp in local_experiments:
        local_path = EXPERIMENTS_DIR / exp / "processed_data"
        local_hash = compute_dir_hash(local_path)

        has_remote = exp in remote_experiments
        if not has_remote:
            to_push.append((exp, "local only"))
        else:
            # Check if ahead
            local_manifest = get_local_manifest(exp)
            if not local_manifest or local_manifest.get("hash") != local_hash:
                to_push.append((exp, "ahead"))
            else:
                # Check remote hash
                try:
                    manifest_path = hf_hub_download(
                        repo_id,
                        f"{exp}/{MANIFEST_FILENAME}",
                        repo_type="dataset",
                        local_dir=REPO_ROOT / ".hf_cache",
                        force_download=True,
                    )
                    with open(manifest_path) as f:
                        remote_manifest = json.load(f)
                    remote_hash = remote_manifest.get("hash", "")
                    if local_hash != remote_hash:
                        to_push.append((exp, "ahead"))
                except Exception:
                    pass  # If we can't check, skip

    if not to_push:
        print("All experiments are synced. Nothing to push.")
        return

    print(f"Found {len(to_push)} experiment(s) to push:")
    for exp, status in to_push:
        print(f"  - {exp} ({status})")
    print()

    # Push each experiment
    failed = []
    for exp, status in to_push:
        local_path = EXPERIMENTS_DIR / exp / "processed_data"
        local_hash = compute_dir_hash(local_path)
        local_size = get_dir_size(local_path)

        print(f"Pushing: {exp}")
        print(f"  Size: {format_size(local_size)}")

        # Create manifest
        manifest = {
            "experiment": exp,
            "hash": local_hash,
            "size_bytes": local_size,
            "pushed_at": datetime.now().isoformat(),
            "files": []
        }

        for file_path in sorted(local_path.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                rel_path = file_path.relative_to(local_path)
                manifest["files"].append({
                    "path": str(rel_path),
                    "size": file_path.stat().st_size
                })

        save_local_manifest(exp, manifest)

        api = HfApi()
        commit_message = f"{exp} @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        try:
            upload_with_retry(
                api,
                folder_path=str(local_path),
                repo_id=repo_id,
                path_in_repo=exp,
                commit_message=commit_message,
                )
            print(f"  ✓ Done\n")
        except Exception as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                print(f"  ? Timed out (may have succeeded)\n")
            else:
                print(f"  ✗ Failed: {e}\n")
                failed.append(exp)

    if failed:
        print(f"\nFailed to push: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All pushes completed successfully!")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Manage experiment data on Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                          Show sync status
  %(prog)s push 03-twenty-questions        Push experiment data
  %(prog)s pushall                         Push all local-only/ahead experiments
  %(prog)s pull 03-twenty-questions        Pull experiment data
  %(prog)s pull 03-twenty-questions/traj   Pull subdirectory
  %(prog)s pull --all                      Pull all experiments
  %(prog)s init                            Initialize HF connection
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize HF repo connection")
    init_parser.set_defaults(func=cmd_init)

    # status
    status_parser = subparsers.add_parser("status", help="Show sync status")
    status_parser.set_defaults(func=cmd_status)

    # push
    push_parser = subparsers.add_parser("push", help="Push experiment to HF")
    push_parser.add_argument("experiment", help="Experiment name to push")
    push_parser.add_argument("--no-xet", action="store_true",
                             help="Disable xet transfer (slower but avoids xet-core panics)")
    push_parser.set_defaults(func=cmd_push)

    # pushall
    pushall_parser = subparsers.add_parser("pushall", help="Push all local-only and ahead experiments")
    pushall_parser.add_argument("--no-xet", action="store_true",
                                help="Disable xet transfer (slower but avoids xet-core panics)")
    pushall_parser.set_defaults(func=cmd_pushall)

    # pull
    pull_parser = subparsers.add_parser("pull", help="Pull from HF")
    pull_parser.add_argument("path", nargs="?", help="Experiment or subpath to pull")
    pull_parser.add_argument("--all", action="store_true", help="Pull all experiments")
    pull_parser.set_defaults(func=cmd_pull)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Must set HF_HUB_DISABLE_XET BEFORE importing huggingface_hub,
    # otherwise the xet native library is already loaded and ignores it.
    if getattr(args, "no_xet", False):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("Xet disabled (--no-xet)")

    args.func(args)


if __name__ == "__main__":
    main()
