#!/usr/bin/env bash
# Submit BOTH the trajectory-sampled (TauColosseum) and IID Colosseum
# benchmarks in parallel via the Snakemake SLURM executor. Each (dim, noise)
# cell becomes one SLURM job; the two pipelines share the same SLURM
# scaffolding but write to disjoint output directories.
#
# 5 dims × 4 noise = 20 cells per pipeline → 40 cells total. With the slurm
# executor running both targets simultaneously and MAX_JOBS large enough,
# everything fans out concurrently.
#
# Run on a Della login node from inside the experiment directory.
#
# Env overrides (defaults match exp11 OOM-resistant config):
#   SLURM_PARTITION   default: cpu
#   SLURM_ACCOUNT     default: henderson
#   CPUS_PER_TASK     default: 8
#   MEM_MB            default: 48000  (raise if cells OOM)
#   RUNTIME_MIN       default: 240    (4h; cells should mostly finish faster)
#   MAX_JOBS          default: 40

set -euo pipefail
cd "$(dirname "$0")"

PARTITION=${SLURM_PARTITION:-cpu}
ACCOUNT=${SLURM_ACCOUNT:-henderson}
CPUS=${CPUS_PER_TASK:-8}
MEM=${MEM_MB:-48000}
RUNTIME=${RUNTIME_MIN:-240}
MAX_JOBS=${MAX_JOBS:-40}

DEFAULT_RES=(
    "slurm_partition=$PARTITION"
    "mem_mb=$MEM"
    "runtime=$RUNTIME"
    "cpus_per_task=$CPUS"
    "slurm_account=$ACCOUNT"
)

CONFIG_OVERRIDES=(
    "cell_mem_mb=$MEM"
    "cell_runtime=$RUNTIME"
    "cell_cpus=$CPUS"
)

mkdir -p logs/tau_slurm logs/iid_slurm

echo "Submitting tau + iid sweeps via Snakemake SLURM executor"
echo "  partition: $PARTITION   account: $ACCOUNT"
echo "  per-cell:  ${CPUS} CPU,  ${MEM} MB,  ${RUNTIME} min"
echo "  max jobs:  $MAX_JOBS    (40 cells total: 20 tau + 20 iid)"

pixi run snakemake \
    --executor slurm \
    -j "$MAX_JOBS" \
    --default-resources "${DEFAULT_RES[@]}" \
    --config "${CONFIG_OVERRIDES[@]}" \
    --keep-going \
    --keep-incomplete \
    --rerun-incomplete \
    figures/v2/tau_colosseum_pearson_compact.png \
    figures/v2/colosseum_pearson_compact_v3.png

echo "Done."
