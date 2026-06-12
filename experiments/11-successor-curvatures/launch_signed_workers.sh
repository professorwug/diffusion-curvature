#!/bin/bash
# Launch the signed-benchmark workers with JAX pinned to CPU.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for w in 0 1 2 3 4 5 6 7; do
  nohup pixi run python benchmark_signed_on_iid.py run --worker-id $w --num-workers 8 \
    > logs/signed_w$w.log 2>&1 &
done
echo launched
