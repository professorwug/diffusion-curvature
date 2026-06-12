#!/usr/bin/env bash
# Launch the full trajectory-sampled Colosseum benchmark across N workers
# on the local box (alternating between cuda:0 and cuda:1).
#
# Usage:
#   ./run_tau_colosseum_sweep.sh [num_workers]
#
# Defaults to 8 workers. Each worker writes its own shard CSV; once they
# all finish the script merges into processed_data/tau_metrics.csv and
# computes the Pearson summary needed by plot_colosseum_pearson_compact.py.

set -euo pipefail

cd "$(dirname "$0")"

NUM_WORKERS=${1:-8}
LOG_DIR=logs/tau_sweep
mkdir -p "$LOG_DIR" processed_data

echo "Launching $NUM_WORKERS workers …"
PIDS=()
for K in $(seq 0 $((NUM_WORKERS - 1))); do
    GPU=$((K % 2))
    LOG="$LOG_DIR/worker_${K}.log"
    echo "  worker $K → cuda:$GPU  (log: $LOG)"
    pixi run python benchmark_all_on_tau_colosseum.py \
        --cmd run \
        --worker-id "$K" \
        --num-workers "$NUM_WORKERS" \
        --device "cuda:$GPU" \
        --out "processed_data/tau_metrics_w${K}.csv" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    # small stagger so workers don't all hit the cache builder simultaneously
    sleep 4
done

echo "PIDs: ${PIDS[*]}"
echo "Waiting for workers …"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  worker pid $pid exited non-zero"
        FAIL=$((FAIL + 1))
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo "WARNING: $FAIL worker(s) failed. Check logs in $LOG_DIR/"
fi

echo "Merging shards …"
pixi run python benchmark_all_on_tau_colosseum.py \
    --cmd merge --num-workers "$NUM_WORKERS"

echo "Building Pearson summary …"
pixi run python summarize_tau_colosseum.py

echo "Done."
echo "Plot the figure with:"
echo "  pixi run python plot_colosseum_pearson_compact.py \\"
echo "    --summary processed_data/v2/tau_colosseum_pearson_summary.csv \\"
echo "    --out figures/v2/tau_colosseum_pearson_compact"
