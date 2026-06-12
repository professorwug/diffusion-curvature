#!/bin/bash
# Live dashboard for benchmark_all_on_iid_v2 workers.
# Refreshes every 15s. Shows per-worker alive-ness, last log line, shard row count.

cd "$(dirname "$0")"

while true; do
  clear
  echo "=== benchmark_all_on_iid_v2 live dashboard  ($(date '+%F %T')) ==="
  echo

  if [ -f logs/bench_v2_pids.txt ]; then
    echo "--- Worker status ---"
    w=0
    while read -r pid; do
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        rss=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ')
        state="ALIVE  pid=$pid  elapsed=$etime  rss=${rss}KB"
      else
        state="EXITED pid=$pid"
      fi
      log="logs/bench_v2_w${w}.log"
      last=""
      if [ -f "$log" ]; then
        last=$(tail -n 1 "$log" 2>/dev/null | cut -c 1-120)
      fi
      printf "  w%d  %s\n         last: %s\n" "$w" "$state" "$last"
      w=$((w + 1))
    done < logs/bench_v2_pids.txt
  else
    echo "  (no PID file yet)"
  fi

  echo
  echo "--- Shard row counts ---"
  shopt -s nullglob
  total=0
  for f in processed_data/iid_metrics_v2_bench_w*.csv; do
    n=$(wc -l < "$f")
    printf "  %s : %d rows\n" "$f" "$n"
    total=$((total + n))
  done
  shopt -u nullglob
  echo "  ------------------------------"
  echo "  TOTAL (incl. headers): $total"

  echo
  echo "--- GPU utilization ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader 2>/dev/null \
      | sed 's/^/  /'

  echo
  echo "(refresh 15s, Ctrl+C to exit dashboard — workers keep running)"
  sleep 15
done
