#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup pixi run python expressiveness_gate.py spectral > logs/express_spectral.log 2>&1 &
nohup pixi run python expressiveness_gate.py run --worker-id 0 --num-workers 2 --device cuda:0 > logs/express_w0.log 2>&1 &
nohup pixi run python expressiveness_gate.py run --worker-id 1 --num-workers 2 --device cuda:1 > logs/express_w1.log 2>&1 &
echo launched
