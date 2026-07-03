#!/bin/bash
cd "$(dirname "$0")"
pkill -f "se_enhanced_colosseum.py run"
sleep 3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for w in 0 1 2 3; do
  dev="cuda:$((w % 2))"
  nohup pixi run python se_enhanced_colosseum.py run --worker-id $w --num-workers 4 --device $dev >> logs/se_enh_w$w.log 2>&1 &
done
echo relaunched
