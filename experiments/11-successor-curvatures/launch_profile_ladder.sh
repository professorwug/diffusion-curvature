#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup pixi run python profile_ladder.py --worker-id 0 --num-workers 2 --device cuda:0 > logs/profile_ladder_w0.log 2>&1 &
nohup pixi run python profile_ladder.py --worker-id 1 --num-workers 2 --device cuda:1 > logs/profile_ladder_w1.log 2>&1 &
echo launched
