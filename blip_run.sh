#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

BATCH=32
SEED=42

source .venv/bin/activate

echo "start blip2"

python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --eval False 2> results/blip2_err

rm -rf ~/.cache/huggingface/hub/*

echo "done blip2"