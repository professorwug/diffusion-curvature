#!/bin/bash
#SBATCH --job-name=exp-template
#SBATCH --account=henderson
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40      # RULE: 8 cores per GPU (5 GPUs × 8 = 40)
#SBATCH --gres=gpu:5
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================
# EXPERIMENT TEMPLATE - Using Della Shared Library
# ============================================================
# Copy this template and customize the EXPERIMENT CONFIG section.
# The shared library handles all vLLM server management.
# ============================================================

set -e

# === EXPERIMENT CONFIG (customize this section) ===
EXPERIMENT_NAME="My Experiment"
WORK_DIR="/scratch/gpfs/HENDERSON/km5839/reason_reckon/experiments/XX-my-experiment"
WORKER_SCRIPT="python run_experiment.py"
NUM_WORKERS=4

# Model config
QWEN_MODEL="/scratch/gpfs/HENDERSON/transformer_cache/Qwen3-Next-80B-A3B-Instruct"
EMBED_MODEL="/scratch/gpfs/HENDERSON/transformer_cache/Llama-3.1-8B-Instruct"

# GPU allocation (pli-c: 5 GPUs)
QWEN_GPUS="0,1,2,3"
QWEN_TP=4
EMBED_GPU="4"

# === END EXPERIMENT CONFIG ===

# Load shared library
cd "$WORK_DIR"
mkdir -p logs
REPO_ROOT="$(dirname "$(dirname "$WORK_DIR")")"
source "$REPO_ROOT/.zetteldev/della/lib/della.sh"

# Job header
log_job_header "$EXPERIMENT_NAME"
echo "Args: $@"

# Setup and launch servers
della_vllm_setup

log_section "Launching vLLM servers"
launch_vllm_server "$QWEN_MODEL" "$QWEN_GPUS" "$QWEN_TP" "$VLLM_PORT"
launch_embedding_server "$EMBED_MODEL" "$EMBED_GPU" "$EMBED_PORT"

log_section "Waiting for servers"
wait_for_servers \
    "http://localhost:$VLLM_PORT|Qwen" \
    "http://localhost:$EMBED_PORT|Embedding"

# Switch to experiment environment
log_section "Running experiment"
della_experiment_env "$WORK_DIR"
export OPENAI_API_BASE="http://localhost:$VLLM_PORT/v1"

# Parse --game-ids from args
GAME_IDS=""
OTHER_ARGS=""
for arg in "$@"; do
    if [[ "$arg" == --game-ids=* ]]; then
        GAME_IDS="${arg#--game-ids=}"
    else
        OTHER_ARGS="$OTHER_ARGS $arg"
    fi
done

# Run workers
run_parallel_workers "$NUM_WORKERS" "$WORKER_SCRIPT" "--game-ids" "$GAME_IDS" \
    --embedding-api-base "http://localhost:$EMBED_PORT/v1" \
    --embedding-model "$EMBED_MODEL" \
    $OTHER_ARGS

log_job_footer
