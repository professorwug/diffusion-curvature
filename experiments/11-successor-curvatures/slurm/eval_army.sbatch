#!/bin/bash
#SBATCH --job-name=dc-eval-army
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:55:00
#SBATCH --output=logs/army_%j.out
# Generic worker-packed evaluation job for the exp-11 benchmark scripts.
# All scripts share the (--worker-id --num-workers --device, resume-CSV)
# pattern, so one job runs WORKERS_PER_GPU x NGPU workers, and NUM_JOBS jobs
# tile the full worker range.
#
# Required env (via sbatch --export):
#   SCRIPT        e.g. diffusing_fraction_colosseum.py
#   SUBCMD        e.g. "run" (some scripts use run-v2 etc.)
#   JOB_INDEX     0..NUM_JOBS-1
#   NUM_JOBS      total jobs tiling the worker range
# Optional:
#   WORKERS_PER_GPU (default 8) | SCRIPT_ARGS (extra args)

set -euo pipefail
cd ~/src/diffusion-curvature/experiments/11-successor-curvatures
PY=../../.pixi/envs/default/bin/python
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=2 \
       XLA_PYTHON_CLIENT_PREALLOCATE=false

NGPU=$(nvidia-smi -L | wc -l)
WPG=${WORKERS_PER_GPU:-8}
WPJ=$((NGPU * WPG))
TOTAL=$((NUM_JOBS * WPJ))
echo "job $JOB_INDEX/$NUM_JOBS: $NGPU GPUs x $WPG workers = $WPJ (total $TOTAL)"
echo "script: $SCRIPT $SUBCMD ${SCRIPT_ARGS:-}"

for k in $(seq 0 $((WPJ - 1))); do
  wid=$((JOB_INDEX * WPJ + k))
  dev="cuda:$((k % NGPU))"
  $PY "$SCRIPT" "$SUBCMD" --worker-id "$wid" --num-workers "$TOTAL" \
      --device "$dev" ${SCRIPT_ARGS:-} \
      > "logs/army_${SLURM_JOB_ID}_w${wid}.log" 2>&1 &
done
wait
echo "job $JOB_INDEX done"
