#!/bin/bash
# Usage (on della login):
#   ./slurm/submit_army.sh <SCRIPT> <SUBCMD> [NUM_JOBS] [GPUS_PER_JOB] [PARTITION] [EXTRA_ARGS...]
# Example:
#   ./slurm/submit_army.sh diffusing_fraction_colosseum.py run 4 4 pli-c
set -euo pipefail
cd "$(dirname "$0")/.."
SCRIPT=$1; SUBCMD=$2
NUM_JOBS=${3:-4}; GPUS=${4:-4}; PARTITION=${5:-pli-c}
shift $(( $# > 5 ? 5 : $# )); EXTRA="${*:-}"
mkdir -p logs
for J in $(seq 0 $((NUM_JOBS - 1))); do
  sbatch --partition="$PARTITION" --gres=gpu:"$GPUS" \
    --export=ALL,SCRIPT="$SCRIPT",SUBCMD="$SUBCMD",JOB_INDEX="$J",NUM_JOBS="$NUM_JOBS",SCRIPT_ARGS="$EXTRA" \
    slurm/eval_army.sbatch
done
squeue -u "$USER" -o "%i %j %T %P %b" | tail -n $((NUM_JOBS + 1))
