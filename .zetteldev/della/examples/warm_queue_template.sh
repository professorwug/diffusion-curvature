#!/bin/bash
# warm_queue_template.sh - Example queue warmer using shared library
#
# Usage:
#   ./warm_queue_template.sh [--dry-run] [--once]

set -e

# === EXPERIMENT CONFIG (customize this section) ===
EXPERIMENT_NAME="exp-template"
WORK_DIR="/scratch/gpfs/HENDERSON/km5839/reason_reckon/experiments/XX-my-experiment"
OUTPUT_DIR="$WORK_DIR/processed_data/output"
TOTAL_ITEMS=50

# Partition configs: name, batch_size, max_jobs, script
# ailab: 12 items per job, max 3 jobs
# pli-c: 4 items per job, max 3 jobs
# === END EXPERIMENT CONFIG ===

# Parse args
DRY_RUN=""
ONCE=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
        --once) ONCE=true ;;
    esac
done

# Load shared library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/della.sh"

# Configure queue
configure_queue \
    "$EXPERIMENT_NAME" \
    "$WORK_DIR" \
    "$OUTPUT_DIR" \
    "$TOTAL_ITEMS" \
    "g%04d" \
    ".json"

# Add partitions
add_partition "ailab" 12 3 "scripts/submit_ailab.sbatch"
add_partition "pli-c" 4 3 "scripts/submit_plic.sbatch"

# Run
if $ONCE; then
    warm_queue_once $DRY_RUN
else
    warm_queue_loop 600  # Check every 10 minutes
fi
