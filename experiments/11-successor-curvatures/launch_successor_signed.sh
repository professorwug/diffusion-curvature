#!/bin/bash
# Two GPU workers for the successor-signed subset (80 instances).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup pixi run python benchmark_successor_signed.py --worker-id 0 --num-workers 2 --device cuda:0 > logs/succ_signed_w0.log 2>&1 &
nohup pixi run python benchmark_successor_signed.py --worker-id 1 --num-workers 2 --device cuda:0 > logs/succ_signed_w1.log 2>&1 &
echo launched
