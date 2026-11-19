#!/bin/bash
#SBATCH --job-name=dc-profile-fleet
#SBATCH --partition=gpu
#SBATCH --account=henderson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/km5839/src/diffusion-curvature/experiments/11-successor-curvatures/logs/profile_fleet_%A_%a.out
# The gESS profile area on the Curvature Colosseum, a fleet of FB nets to a card.
# One net cannot fill a card, so FLEET of them train in one batched pass: every pending
# manifold of a (regime, dimension) group at once. NUM_TASKS array tasks tile the ten
# groups (two regimes by five dimensions). A task appends to its own shard after every
# fleet and stops before the wall, so a resubmission of the same array resumes.
#
# Submit through the reason_reckon gate on the login node, never by raw sbatch. The
# first run of a new layout is one task with --smoke:
#   ~/src/reason_reckon/.zetteldev/della/bin/della-submit --no-wait --array 0-<NUM_TASKS-1> \
#       --env NUM_TASKS=<NUM_TASKS> ~/src/diffusion-curvature/experiments/11-successor-curvatures/slurm/profile_area_gpu.sbatch
#
# preflight: cd .. && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ../../.pixi/envs/default/bin/python -c "import numpy as np, benchmark_profile_area as b; from diffusion_curvature.successor.ensemble import FleetFBTrainer as F; t = F(obs_dim=3, seeds=[1, 2], n_epochs=1, device='cpu', **b.FB_BASE).fit([np.random.rand(4, 9, 3).astype('float32')] * 2); print('fleet of 2 fits:', t.measures(0, np.zeros((5, 3)), np.zeros((5, 3)))[0].shape)"
set -euo pipefail
cd "$HOME/src/diffusion-curvature/experiments/11-successor-curvatures"
PY=../../.pixi/envs/default/bin/python
NUM_TASKS=${NUM_TASKS:?set NUM_TASKS to the size of the array}; TASK=${SLURM_ARRAY_TASK_ID:-0}
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs processed_data
echo "task $TASK of $NUM_TASKS: fleets of ${FLEET:-200}, commit ${REPO_COMMIT:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u benchmark_profile_area.py fleet --task-id "$TASK" --num-tasks "$NUM_TASKS" --device cuda --fleet "${FLEET:-200}" \
    --prep-workers 8 --max-minutes "${MAX_MINUTES:-52}" ${SCRIPT_ARGS:-}
echo "task $TASK done"
