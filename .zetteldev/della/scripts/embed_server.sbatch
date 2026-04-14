#!/bin/bash
#SBATCH --job-name=embed-server-km5839
#SBATCH --account=henderson
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # RULE: 8 cores per GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/embed_server_%j.out
#SBATCH --error=logs/embed_server_%j.err

# ============================================================
# SHARED EMBEDDING SERVER
# ============================================================
# Long-running embedding server that production jobs can share.
# Production jobs discover this via squeue and connect via SSH tunnel.
#
# Start this ONCE before running production jobs:
#   sbatch .zetteldev/della/scripts/embed_server.sbatch
#
# Check status:
#   squeue -u $USER --name=embed-server-$USER
#
# Production jobs automatically connect using:
#   connect_shared_embedding  # from della library
# ============================================================

set -e

# Fixed port for discovery
EMBED_PORT=12851
SERVED_NAME="embedding"

# Model for embeddings
EMBED_MODEL="/scratch/gpfs/HENDERSON/transformer_cache/Llama-3.1-8B-Instruct"

# Setup
REPO_ROOT="/scratch/gpfs/HENDERSON/km5839/reason_reckon"
cd "$REPO_ROOT"
mkdir -p logs

source "$REPO_ROOT/.zetteldev/della/lib/della.sh"

log_job_header "Shared Embedding Server"
echo "Port: $EMBED_PORT"
echo "Served as: $SERVED_NAME"
echo "Model: $EMBED_MODEL"
echo ""
echo "Production jobs can connect using:"
echo "  connect_shared_embedding"
echo ""

# Setup cleanup and vLLM environment
setup_cleanup_trap
kill_orphaned_vllm
setup_vllm_env

# Launch embedding server on fixed port
log_section "Launching embedding server"
launch_embedding_server "$EMBED_MODEL" "0" "$EMBED_PORT" "$SERVED_NAME"

# Wait for server to be ready
wait_for_server "http://localhost:$EMBED_PORT" "Embedding" 10

log_section "Server ready"
echo "Embedding server ready at http://localhost:$EMBED_PORT"
echo "Serving model as: $SERVED_NAME"
echo ""
echo "Production jobs on other nodes can connect via SSH tunnel."
echo "The della library handles this automatically."
echo ""

# Keep server running until job timeout or cancellation
# Sleep in chunks to allow for signal handling
log_section "Running until timeout or cancellation"
while true; do
    sleep 300
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server still running..."

    # Health check
    if ! curl -s "http://localhost:$EMBED_PORT/v1/models" > /dev/null 2>&1; then
        echo "WARNING: Server health check failed!"
    fi
done
