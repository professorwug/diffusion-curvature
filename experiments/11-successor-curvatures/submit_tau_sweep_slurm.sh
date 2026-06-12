#!/usr/bin/env bash
# Submit the trajectory-sampled Colosseum benchmark via Snakemake's SLURM
# executor. Each (dim, noise) cell becomes one SLURM job; cells run
# concurrently up to the cluster's queue limit.
#
# Run this on a login node from inside the experiment directory.
#
# Env overrides:
#   SLURM_PARTITION   default: cpu
#   SLURM_ACCOUNT     default: (unset; set if your cluster requires it)
#   CPUS_PER_TASK     default: 8
#   MEM_MB            default: 16000
#   RUNTIME_MIN       default: 180
#   MAX_JOBS          default: 20  (one per cell; raise if you've added cells)
#
# Example:
#   SLURM_PARTITION=cpu SLURM_ACCOUNT=henderson CPUS_PER_TASK=8 \
#       ./submit_tau_sweep_slurm.sh

set -euo pipefail
cd "$(dirname "$0")"

PARTITION=${SLURM_PARTITION:-cpu}
ACCOUNT=${SLURM_ACCOUNT:-}
CPUS=${CPUS_PER_TASK:-8}
MEM=${MEM_MB:-16000}
RUNTIME=${RUNTIME_MIN:-180}
MAX_JOBS=${MAX_JOBS:-20}

DEFAULT_RES=(
    "slurm_partition=$PARTITION"
    "mem_mb=$MEM"
    "runtime=$RUNTIME"
    "cpus_per_task=$CPUS"
)
if [[ -n "$ACCOUNT" ]]; then
    DEFAULT_RES+=("slurm_account=$ACCOUNT")
fi

mkdir -p logs/tau_slurm

echo "Submitting tau-colosseum sweep via Snakemake SLURM executor"
echo "  partition: $PARTITION"
echo "  account:   ${ACCOUNT:-<none>}"
echo "  cpus:      $CPUS"
echo "  mem:       ${MEM} MB"
echo "  runtime:   ${RUNTIME} min"
echo "  max jobs:  $MAX_JOBS"

pixi run snakemake \
    --executor slurm \
    -j "$MAX_JOBS" \
    --default-resources "${DEFAULT_RES[@]}" \
    --keep-going \
    --rerun-incomplete \
    figures/v2/tau_colosseum_pearson_compact.png

echo "Done."
