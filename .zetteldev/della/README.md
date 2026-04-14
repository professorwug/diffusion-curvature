# Della Cluster Utilities

Shared bash library for running vLLM experiments on the Della cluster.

## Design Philosophy

**ALWAYS use these library functions instead of manually launching vLLM servers.** The library provides:
- Automatic cleanup of orphaned processes
- Unique port assignment to avoid conflicts
- Consistent environment configuration
- PID tracking and trap handling
- Less boilerplate code (~60 lines → ~10 lines)

## CPU Core Allocation Rule

**CRITICAL: Always allocate 8 CPU cores per GPU** in your SBATCH directives.

Each ailab node has 64 CPU cores and 8 GPUs. Following the 8 cores/GPU rule ensures:
- Efficient CPU utilization (target: >90% CPU efficiency)
- Maximum cluster throughput (no GPU starvation due to CPU exhaustion)
- Fair resource sharing across jobs

### Standard Allocations

| GPUs | CPU Cores | Example |
|------|-----------|---------|
| 1    | 8         | `#SBATCH --cpus-per-task=8` `#SBATCH --gres=gpu:1` |
| 2    | 16        | `#SBATCH --cpus-per-task=16` `#SBATCH --gres=gpu:2` |
| 3    | 24        | `#SBATCH --cpus-per-task=24` `#SBATCH --gres=gpu:3` |
| 4    | 32        | `#SBATCH --cpus-per-task=32` `#SBATCH --gres=gpu:4` |
| 5    | 40        | `#SBATCH --cpus-per-task=40` `#SBATCH --gres=gpu:5` |
| 8    | 64        | `#SBATCH --cpus-per-task=64` `#SBATCH --gres=gpu:8` |

### Monitoring CPU Efficiency

After submitting a job, check CPU efficiency:
```bash
seff <JOBID>  # Should show >70% CPU efficiency
```

If CPU efficiency is consistently >95%, you may be CPU-bound and could benefit from slight increases. If <70%, you're over-allocated and should reduce cores.

## Quick Start

```bash
# In your sbatch script:
REPO_ROOT="/scratch/gpfs/HENDERSON/km5839/reason_reckon"
source "$REPO_ROOT/.zetteldev/della/lib/della.sh"

# Setup vLLM environment (cleanup, env vars, ports)
della_vllm_setup

# Launch servers (tool-call parser auto-detected from model name)
launch_vllm_server "$MODEL" "0,1,2,3" 4 "$VLLM_PORT" "model"

wait_for_server "http://localhost:$VLLM_PORT" "Model" 60

# Setup experiment environment (uv, PYTHONPATH, tokens)
della_experiment_env "$WORK_DIR"
export OPENAI_API_BASE="http://localhost:$VLLM_PORT/v1"

# Run your experiment
python run_experiment.py --model "openai/model" ...
```

### What NOT to Do ❌

```bash
# DON'T manually launch servers like this:
source "$VLLM_ENV"
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/gpfs/HENDERSON/transformer_cache
LLAMA_PORT=$((10000 + (SLURM_JOB_ID % 5000)))

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$LLAMA_MODEL" \
    --tensor-parallel-size 4 \
    --port $LLAMA_PORT \
    --enforce-eager \
    ...
    > "logs/llama_${SLURM_JOB_ID}.log" 2>&1 &
LLAMA_PID=$!

cleanup() {
    kill $LLAMA_PID 2>/dev/null || true
    wait $LLAMA_PID 2>/dev/null || true
}
trap cleanup EXIT
# ... 50+ more lines of boilerplate
```

### Do This Instead ✅

```bash
# Use the library - much cleaner!
source "$REPO_ROOT/.zetteldev/della/lib/della.sh"
della_vllm_setup
launch_vllm_server "$LLAMA_MODEL" "0,1,2,3" 4 "$VLLM_PORT" "llama"
# Tool-call parser auto-detected as llama3_json
wait_for_server "http://localhost:$VLLM_PORT" "Llama" 60
della_experiment_env "$WORK_DIR"
```

## Library Modules

| Module | Description |
|--------|-------------|
| `della.sh` | Main entry point, sources all modules |
| `logging.sh` | Formatted logging (headers, colors, sections) |
| `port_utils.sh` | Unique port generation to avoid conflicts |
| `cleanup.sh` | Process cleanup and trap handlers |
| `vllm_server.sh` | vLLM server lifecycle (launch, wait, configure) |
| `shared_embed.sh` | Shared embedding server discovery and tunneling |
| `worker.sh` | Parallel worker execution |
| `queue.sh` | Job queue management (warm queue pattern) |

## Key Functions

### della.sh (High-level helpers)

```bash
della_vllm_setup                          # Full vLLM setup: cleanup, env, ports
della_experiment_env [WORK_DIR]           # Switch to UV env, set PYTHONPATH, tokens
```

### vllm_server.sh

```bash
setup_vllm_env                            # Configure HF_HOME, activate venv
launch_vllm_server MODEL GPUS TP PORT [SERVED_NAME] [EXTRA_ARGS...]
launch_embedding_server MODEL GPU PORT [SERVED_NAME] [EXTRA_ARGS...]
wait_for_server URL NAME [TIMEOUT_MIN]
wait_for_servers "URL|Name" "URL|Name" ...
```

**Note:** `SERVED_NAME` defaults to `"model"` for vLLM servers. Pass extra args like `--chat-template`, etc. after the base parameters.

### Tool Calling (Auto-Detection)

The library **automatically detects** the correct tool-call parser based on the model name:

| Model Family | Parser |
|--------------|--------|
| Qwen         | `hermes` |
| Llama        | `llama3_json` |
| Mistral      | `mistral` |

This happens automatically - no need to specify `--tool-call-parser` in most cases:

```bash
# Qwen model → auto-detects hermes parser
launch_vllm_server "$QWEN_MODEL" "0,1,2,3" 4 "$VLLM_PORT" "model"
#   Tool calling: auto-detected parser 'hermes'

# Llama model → auto-detects llama3_json parser
launch_vllm_server "$LLAMA_MODEL" "0,1,2,3" 4 "$VLLM_PORT" "model"
#   Tool calling: auto-detected parser 'llama3_json'
```

**Override:** Pass `--tool-call-parser` explicitly to use a different parser:
```bash
launch_vllm_server "$MODEL" "0,1,2,3" 4 "$VLLM_PORT" "model" \
    --tool-call-parser mistral
```

**Opt-out:** Pass `--no-tool-calling` to disable tool calling entirely:
```bash
launch_vllm_server "$MODEL" "0,1,2,3" 4 "$VLLM_PORT" "model" \
    --no-tool-calling
```

### Common Customizations

```bash
# Add custom chat template for Llama models
launch_vllm_server "$LLAMA_MODEL" "0,1,2,3" 4 "$VLLM_PORT" "llama" \
    --chat-template "$WORK_DIR/scripts/tool_chat_template_llama3.1_json.jinja"

# Adjust model context length
launch_vllm_server "$MODEL" "0,1" 2 "$VLLM_PORT" "model" \
    --max-model-len 16384

# Change GPU memory utilization
launch_vllm_server "$MODEL" "0,1" 2 "$VLLM_PORT" "model" \
    --gpu-memory-utilization 0.95

# Disable tool calling for embedding-only or completion models
launch_vllm_server "$MODEL" "0" 1 "$VLLM_PORT" "model" \
    --no-tool-calling
```

The library sets sensible defaults (`--enforce-eager`, `--gpu-memory-utilization 0.90`) and auto-detects tool calling settings. Any default can be overridden by passing explicit args.

### worker.sh

```bash
run_parallel_workers NUM CMD FLAG ITEMS [ARGS...]
# Example: run_parallel_workers 4 "python run.py" "--ids" "a,b,c,d"
```

### queue.sh

```bash
configure_queue JOB_NAME WORK_DIR OUTPUT_DIR TOTAL [PATTERN] [EXT]
add_partition NAME BATCH_SIZE MAX_JOBS SCRIPT
warm_queue_once [--dry-run]
warm_queue_loop [INTERVAL_SECONDS]
```

### port_utils.sh

```bash
eval "$(get_vllm_ports)"   # Sets VLLM_PORT, EMBED_PORT
port=$(get_random_port 20000 25000)
```

### shared_embed.sh

```bash
# Check if shared server is running
shared_embed_available

# Find the node running the server
node=$(find_embed_server_node)

# Connect to shared server (fail if unavailable)
connect_shared_embedding

# Connect with fallback to local server
connect_shared_embedding "$LLAMA_MODEL" "2"

# Wait for server to start (with timeout)
wait_for_shared_embed 300  # 5 minutes

# Get status information
shared_embed_status
```

## Shared Embedding Server

The shared embedding server eliminates per-job startup overhead by running a single long-lived embedding server that production jobs discover and connect to via SSH tunneling.

### Architecture

```
┌────────────────────────────────────────────────┐
│  Embedding Server Job (embed-server-km5839)    │
│  Node: della-l40s-01                           │
│  Port: 12851 (fixed)                           │
│  Duration: 12 hours                            │
└────────────────────────────────────────────────┘
                    │ SSH Tunnel
                    ▼
┌────────────────────────────────────────────────┐
│  Production Jobs                               │
│  squeue → find node → ssh -L 12851:...:12851   │
│  Use: http://localhost:12851/v1                │
└────────────────────────────────────────────────┘
```

### Quick Start

**1. Start the shared server (once per session):**
```bash
sbatch .zetteldev/della/scripts/embed_server.sbatch
```

**2. In production scripts, use `connect_shared_embedding`:**
```bash
# Require shared server (fail if unavailable)
connect_shared_embedding
export EMBED_API_BASE="http://localhost:$EMBED_PORT/v1"

# OR: Use shared with fallback to local
connect_shared_embedding "$LLAMA_MODEL" "2"
```

### Migration from `launch_embedding_server`

**Before (per-job embedding server):**
```bash
launch_embedding_server "$LLAMA_MODEL" "2" "$EMBED_PORT" "embedding"
wait_for_server "http://localhost:$EMBED_PORT" "Embed" 10
```

**After (shared embedding server):**
```bash
connect_shared_embedding "$LLAMA_MODEL" "2"
# EMBED_PORT is set automatically
# Server wait is handled internally
```

### Benefits

| Aspect | Per-Job Server | Shared Server |
|--------|---------------|---------------|
| Startup time | 2-3 min per job | ~5 sec (tunnel only) |
| GPU usage | 1 GPU per job | 1 GPU total |
| Reliability | Fresh each job | Stable, long-running |

### Monitoring

```bash
# Check if server is running
squeue -u $USER --name=embed-server-$USER

# Detailed status
source .zetteldev/della/lib/della.sh
shared_embed_status

# View server logs
tail -f logs/embed_server_*.out
```

### Troubleshooting

**Server not found:**
```bash
# Start the server
sbatch .zetteldev/della/scripts/embed_server.sbatch

# Wait for it to be ready
wait_for_shared_embed 300
```

**Tunnel connection failed:**
- Check if server node is reachable: `ssh <node> hostname`
- Verify server is running: `curl http://<node>:12851/v1/models`
- Check for port conflicts on local node

**Embeddings not working:**
- Verify model name matches config: shared server uses `"embedding"` as served name
- Check config uses `embedding_model: embedding` (not model path)

## Examples

### Templates (in `examples/` directory)
- `experiment_template.sbatch` - Full experiment job with dual-server setup
- `warm_queue_template.sh` - Queue warmer template

### Real-World Examples
See these production scripts for reference:
- **Single model**: `experiments/10-reflexion-rejection/scripts/submit_batch_baseline_llama70b.sbatch`
  - Llama 3.3 70B with TP=4
  - Custom tool parser and chat template
  - 12 parallel workers
- **Dual models**: `experiments/10-reflexion-rejection/scripts/submit_batch_v4.sbatch`
  - Qwen (TP=2) + Llama embedding server
  - Multi-model orchestration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DELLA_SCRATCH` | `/scratch/gpfs/HENDERSON` | Scratch directory |
| `DELLA_CACHE` | `$DELLA_SCRATCH/transformer_cache` | HuggingFace cache |
| `DELLA_VLLM_ENV` | `/scratch/gpfs/km5839/environments/gpt-oss/bin/activate` | vLLM Python env |
| `DELLA_UV_ENV` | `$DELLA_SCRATCH/km5839/reason_reckon/.venv/bin/activate` | UV Python env |
| `QWEN_MODEL_PATH` | `$DELLA_CACHE/Qwen3-Next-80B-A3B-Instruct` | Default Qwen model |
| `LLAMA_8B_PATH` | `$DELLA_CACHE/Llama-3.1-8B-Instruct` | Default Llama model |
