#!/bin/bash
# Launch 8 shard workers (4 on cuda:0, 4 on cuda:1) for benchmark_all_on_iid_v2.
# Writes per-worker logs; use dashboard_bench_v2.sh to monitor.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

NUM_WORKERS=8
PIDS=()

echo "Starting $NUM_WORKERS workers …"
for w in 0 1 2 3; do
  nohup pixi run python -u benchmark_all_on_iid_v2.py run \
      --worker-id "$w" --num-workers "$NUM_WORKERS" --device cuda:0 \
      > "logs/bench_v2_w${w}.log" 2>&1 &
  PIDS+=("$!")
  echo "  worker $w (cuda:0) PID=$!"
done
for w in 4 5 6 7; do
  nohup pixi run python -u benchmark_all_on_iid_v2.py run \
      --worker-id "$w" --num-workers "$NUM_WORKERS" --device cuda:1 \
      > "logs/bench_v2_w${w}.log" 2>&1 &
  PIDS+=("$!")
  echo "  worker $w (cuda:1) PID=$!"
done

# Persist PIDs so the dashboard can check alive-ness.
printf "%s\n" "${PIDS[@]}" > logs/bench_v2_pids.txt
echo
echo "All workers launched. PIDs saved to logs/bench_v2_pids.txt."
echo "Starting dashboard (Ctrl+C to exit; workers keep running)…"
sleep 3
exec bash dashboard_bench_v2.sh
