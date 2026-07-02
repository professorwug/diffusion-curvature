#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup pixi run python fidelity_gate.py --worker-id 0 --num-workers 2 --device cuda:0 > logs/fidelity_w0.log 2>&1 &
nohup pixi run python fidelity_gate.py --worker-id 1 --num-workers 2 --device cuda:1 > logs/fidelity_w1.log 2>&1 &
echo launched
