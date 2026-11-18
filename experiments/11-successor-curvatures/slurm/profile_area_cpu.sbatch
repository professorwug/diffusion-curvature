#!/bin/bash
#SBATCH --job-name=dc-profile-area
#SBATCH --partition=cpu
#SBATCH --account=henderson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/km5839/src/diffusion-curvature/experiments/11-successor-curvatures/logs/profile_area_%A_%a.out
# The gESS profile area on the Curvature Colosseum, iid and trajectory-sampled, on CPU:
# the FB nets are too small to fill a card. One array task packs WORKERS workers of
# THREADS threads each; NUM_TASKS tasks tile the worker range. Every worker appends to
# its own shard and stops before the wall, so a resubmission of the same array resumes.
#
# Submit through the reason_reckon gate on the login node, never by raw sbatch:
#   ~/src/reason_reckon/.zetteldev/della/bin/della-submit --no-wait --array 0-<NUM_TASKS-1> \
#       --env NUM_TASKS=<NUM_TASKS> ~/src/diffusion-curvature/experiments/11-successor-curvatures/slurm/profile_area_cpu.sbatch
#
# preflight: cd .. && OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= ../../.pixi/envs/default/bin/python benchmark_profile_area.py run --test-run --threads 2 --out-dir /tmp/dc_profile_area_preflight_$USER
set -euo pipefail
cd "$HOME/src/diffusion-curvature/experiments/11-successor-curvatures"
PY=../../.pixi/envs/default/bin/python
WORKERS=${WORKERS:-16}; THREADS=${THREADS:-2}; NUM_TASKS=${NUM_TASKS:?set NUM_TASKS to the size of the array}
TASK=${SLURM_ARRAY_TASK_ID:-0}; TOTAL=$((NUM_TASKS * WORKERS))
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=""
mkdir -p logs processed_data
echo "task $TASK of $NUM_TASKS: $WORKERS workers x $THREADS threads, $TOTAL workers in all, commit ${REPO_COMMIT:-unknown}"
for k in $(seq 0 $((WORKERS - 1))); do
  wid=$((TASK * WORKERS + k))
  $PY -u benchmark_profile_area.py run --worker-id "$wid" --num-workers "$TOTAL" --device cpu --threads "$THREADS" \
      --max-minutes "${MAX_MINUTES:-84}" ${SCRIPT_ARGS:-} > "logs/profile_area_${SLURM_JOB_ID:-local}_w${wid}.log" 2> "logs/profile_area_${SLURM_JOB_ID:-local}_w${wid}.err" &
done
wait
echo "task $TASK done"
