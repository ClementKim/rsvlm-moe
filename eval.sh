#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 baseline/baseline_main.py --model qwen --param 3 --batch 16 --seed 42 --eval True > results/qwen_3_eval

python3 baseline/baseline_main.py --model gemma --param 4 --batch 16 --seed 42 --eval True > results/gemma_4_eval

python3 baseline/baseline_main.py --model blip2 --batch 16 --seed 42 --eval True > results/blip2_eval
