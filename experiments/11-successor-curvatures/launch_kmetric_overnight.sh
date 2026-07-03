#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for w in 0 1 2 3 4 5; do
  dev="cuda:$((w % 2))"
  nohup pixi run python benchmark_kmetric_colosseum.py run --worker-id $w --num-workers 6 --device $dev > logs/kmetric_w$w.log 2>&1 &
done
echo launched
