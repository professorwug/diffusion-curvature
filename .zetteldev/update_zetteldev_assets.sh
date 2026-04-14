#!/usr/bin/env bash
# Update local zetteldev assets from the upstream WherewithAI/zetteldev repo.

set -euo pipefail

DEFAULT_REPO_URL="https://github.com/WherewithAI/zetteldev.git"
DEFAULT_REF="main"

usage() {
  cat <<'EOF'
Usage: update_zetteldev_assets.sh [--repo <url>] [--ref <name>] [--clean]

Sync the zetteldev assets in this repository with the upstream WherewithAI/zetteldev repo.

Options:
  --repo <url>   Override the repository URL to pull assets from.
  --ref <name>   Use a specific git ref (branch, tag, or commit). Defaults to main.
  --clean        Remove local files that are not present upstream when syncing directories.
  -h, --help     Show this help message.

Environment overrides:
  ZETTELDEV_REPO_URL   Same as --repo.
  ZETTELDEV_REF        Same as --ref.
  ZETTELDEV_ASSETS     Comma-separated override for the list of assets to sync.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' is not installed" >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  echo "Error: unable to locate repository root (missing .git directory)." >&2
  echo "Run this script from within a git repository created from the zetteldev template." >&2
  exit 1
fi

REPO_URL="${ZETTELDEV_REPO_URL:-$DEFAULT_REPO_URL}"
REF="${ZETTELDEV_REF:-$DEFAULT_REF}"
CLEAN="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -lt 2 ]] && { echo "Error: --repo requires a value." >&2; exit 1; }
      REPO_URL="$2"
      shift 2
      ;;
    --ref|--branch)
      [[ $# -lt 2 ]] && { echo "Error: $1 requires a value." >&2; exit 1; }
      REF="$2"
      shift 2
      ;;
    --clean)
      CLEAN="yes"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'." >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a ASSET_LIST <<< "${ZETTELDEV_ASSETS:-.zetteldev,.devcontainer,CLAUDE.md,AGENTS.md,infra/infra.md,gambols/gambols.md,experiments/experiments.md,infra,gambols,experiments}"

require_cmd git

TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t zetteldev-sync)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

echo "Fetching ${REPO_URL}@${REF}..."
git clone --depth 1 --branch "$REF" "$REPO_URL" "$TMP_DIR" >/dev/null 2>&1 || {
  echo "Error: failed to clone ${REPO_URL} (ref ${REF})." >&2
  exit 1
}

SOURCE_ROOT="${ZETTELDEV_SOURCE_SUBDIR:+$TMP_DIR/$ZETTELDEV_SOURCE_SUBDIR}"
if [[ -z "$SOURCE_ROOT" ]]; then
  SOURCE_ROOT="$TMP_DIR"
  if [[ ! -e "$SOURCE_ROOT/.zetteldev" ]]; then
    candidate="$(find "$TMP_DIR" -maxdepth 3 -type d -path '*/.zetteldev' -print -quit 2>/dev/null || true)"
    if [[ -n "$candidate" ]]; then
      SOURCE_ROOT="$(dirname "$candidate")"
    fi
  fi
fi

if command -v rsync >/dev/null 2>&1; then
  SYNC_TOOL="rsync"
else
  SYNC_TOOL="tar"
fi

sync_directory() {
  local source_dir="$1"
  local dest_dir="$2"

  mkdir -p "$dest_dir"

  if [[ "$SYNC_TOOL" == "rsync" ]]; then
    local rsync_opts=(-a)
    if [[ "$CLEAN" == "yes" ]]; then
      rsync_opts+=(--delete)
    fi
    rsync "${rsync_opts[@]}" "$source_dir/" "$dest_dir/"
  else
    # tar-based fallback preserves timestamps/permissions but cannot delete extras.
    if [[ "$CLEAN" == "yes" ]]; then
      echo "Warning: --clean requested but rsync is unavailable; skipping cleanup for $dest_dir." >&2
    fi
    (cd "$source_dir" && tar cf - .) | (cd "$dest_dir" && tar xf -)
  fi
}

sync_file() {
  local source_file="$1"
  local dest_file="$2"

  mkdir -p "$(dirname -- "$dest_file")"
  cp "$source_file" "$dest_file"
}

pushd "$REPO_ROOT" >/dev/null

for asset in "${ASSET_LIST[@]}"; do
  asset="${asset//[[:space:]]/}"
  [[ -z "$asset" ]] && continue

  source_path="$SOURCE_ROOT/$asset"
  dest_path="$REPO_ROOT/$asset"

  if [[ ! -e "$source_path" ]]; then
    echo "Warning: asset '$asset' does not exist in upstream; skipping." >&2
    continue
  fi

  if [[ -d "$source_path" ]]; then
    echo "Syncing directory $asset"
    sync_directory "$source_path" "$dest_path"
  else
    echo "Syncing file $asset"
    sync_file "$source_path" "$dest_path"
  fi
done

popd >/dev/null

echo "Done. Remember to review and commit any changes."
