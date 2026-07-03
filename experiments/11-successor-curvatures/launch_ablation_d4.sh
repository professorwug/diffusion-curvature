#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup pixi run python fb_kernel_ablation_d4.py run --worker-id 0 --num-workers 2 --device cuda:0 > logs/abl_d4_w0.log 2>&1 &
nohup pixi run python fb_kernel_ablation_d4.py run --worker-id 1 --num-workers 2 --device cuda:0 > logs/abl_d4_w1.log 2>&1 &
echo launched
