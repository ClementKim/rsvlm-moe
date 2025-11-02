#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH=32
SEED=42

source .venv/bin/activate

# Qwen
echo "start Qwen"

python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --eval False 2> results/qwen_3_err

rm -rf ~/.cache/huggingface/hub/*

python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --eval False 2> results/qwen_7_err

rm -rf ~/.cache/huggingface/hub/*

# python3 baseline/baseline_main.py --model qwen --param 32 --batch $BATCH --seed $SEED --eval False 2> results/qwen_32_err

# rm -rf ~/.cache/huggingface/hub/*

# python3 baseline/baseline_main.py --model qwen --param 72 --batch $BATCH --seed $SEED --eval False 2> results/qwen_72_err

# rm -rf ~/.cache/huggingface/hub/*

echo "done Qwen"

# Llama
# echo "start llama"

# python3 baseline/baseline_main.py --model llama --param 11 --batch $BATCH --seed $SEED --eval False 2> results/llama_11_err

# rm -rf ~/.cache/huggingface/hub/*

# python3 baseline/baseline_main.py --model llama --param 90 --batch $BATCH --seed $SEED --eval False 2> results/llama_90_err

# rm -rf ~/.cache/huggingface/hub/*

# echo "done llama"

# Gemma
echo "start gemma"

python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --eval False 2> results/gemma_4_err

rm -rf ~/.cache/huggingface/hub/*

python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --eval False 2> results/gemma_12_err

rm -rf ~/.cache/huggingface/hub/*

# python3 baseline/baseline_main.py --model gemma --param 27 --batch $BATCH --seed $SEED --eval False 2> results/gemma_27_err

# rm -rf ~/.cache/huggingface/hub/*

echo "done gemma"