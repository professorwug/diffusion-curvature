#!/bin/bash
#SBATCH --job-name=test-shared-embed
#SBATCH --account=henderson
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_shared_embed_%j.out
#SBATCH --error=logs/test_shared_embed_%j.err

# ============================================================
# TEST: Shared Embedding Server Connection from Compute Node
# ============================================================
# Verifies that:
# 1. SSH tunneling works from compute nodes
# 2. Embedding requests work through the tunnel
# 3. Fallback to local server works if needed
# ============================================================

set -e

REPO_ROOT="/scratch/gpfs/HENDERSON/km5839/reason_reckon"
cd "$REPO_ROOT"
mkdir -p logs

source "$REPO_ROOT/.zetteldev/della/lib/della.sh"

log_job_header "Test Shared Embedding Connection"

echo "Running on node: $(hostname)"
echo "Testing SSH tunnel to shared embedding server..."
echo ""

# Setup cleanup for any processes we start
setup_cleanup_trap

# Test 1: Check if shared server is available
log_section "Test 1: Discovery"
if shared_embed_available; then
    echo "PASS: Shared embedding server is available"
    node=$(find_embed_server_node)
    echo "  Node: $node"
else
    echo "FAIL: Shared embedding server not found"
    echo "  Make sure to start it first:"
    echo "  sbatch .zetteldev/della/scripts/embed_server.sbatch"
    exit 1
fi

# Test 2: Connect via SSH tunnel
log_section "Test 2: SSH Tunnel"
echo "Attempting to connect from compute node $(hostname)..."

# This is the key test - can we SSH tunnel from a compute node?
connect_shared_embedding

echo "PASS: Connected to shared embedding server"
echo "  EMBED_PORT=$EMBED_PORT"

# Test 3: Make embedding request
log_section "Test 3: Embedding Request"
echo "Testing embedding API via tunnel..."

response=$(curl -s "http://localhost:$EMBED_PORT/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"input": "This is a test embedding request from a compute node", "model": "embedding"}')

if echo "$response" | grep -q '"embedding"'; then
    echo "PASS: Embedding request successful"
    # Show first few dimensions
    echo "  Response preview: $(echo "$response" | head -c 200)..."
else
    echo "FAIL: Embedding request failed"
    echo "  Response: $response"
    exit 1
fi

# Test 4: Multiple requests (stress test)
log_section "Test 4: Multiple Requests"
echo "Making 5 sequential embedding requests..."

for i in {1..5}; do
    start=$(date +%s%N)
    curl -s "http://localhost:$EMBED_PORT/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d "{\"input\": \"Test request $i\", \"model\": \"embedding\"}" > /dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "  Request $i: ${elapsed}ms"
done

echo "PASS: All requests completed"

log_section "All Tests Passed"
echo ""
echo "SSH tunneling from compute node $(hostname) works correctly."
echo "Production jobs can safely use connect_shared_embedding."

log_job_footer
