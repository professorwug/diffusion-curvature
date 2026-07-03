#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup pixi run python successor_ricci_ladder.py run --worker-id 0 --num-workers 2 --device cuda:0 > logs/sricci_w0.log 2>&1 &
nohup pixi run python successor_ricci_ladder.py run --worker-id 1 --num-workers 2 --device cuda:1 > logs/sricci_w1.log 2>&1 &
for w in 0 1 2 3; do
  dev="cuda:$((w % 2))"
  nohup pixi run python se_enhanced_colosseum.py run --worker-id $w --num-workers 4 --device $dev > logs/se_enh_w$w.log 2>&1 &
done
echo launched
