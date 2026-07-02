#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup pixi run python fb_kernel_measure_ablation.py --worker-id 0 --num-workers 2 --device cuda:0 > logs/measure_abl_w0.log 2>&1 &
nohup pixi run python fb_kernel_measure_ablation.py --worker-id 1 --num-workers 2 --device cuda:1 > logs/measure_abl_w1.log 2>&1 &
echo launched
