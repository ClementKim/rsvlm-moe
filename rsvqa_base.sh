#!/bin/bash

# Set CUDA devices and PyTorch configuration
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
source .venv/bin/activate # venv environment
# conda activate rsvlm # conda environment

# Set batch size and seed
BATCH=32
SEED=42

# Qwen
echo "start Qwen"

# Qwen VL 3B
python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --eval False 2> results/qwen_3_err
rm -rf ~/.cache/huggingface/hub/*

# Qwen VL 7B
python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --eval False 2> results/qwen_7_err
rm -rf ~/.cache/huggingface/hub/*

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

# Gemma 4B
python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --eval False 2> results/gemma_4_err
rm -rf ~/.cache/huggingface/hub/*

# Gemma 12B
python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --eval False 2> results/gemma_12_err
rm -rf ~/.cache/huggingface/hub/*

# Gemma 27B
# python3 baseline/baseline_main.py --model gemma --param 27 --batch $BATCH --seed $SEED --eval False 2> results/gemma_27_err
# rm -rf ~/.cache/huggingface/hub/*

echo "done gemma"

# Blip2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "start blip2"

python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --eval False 2> results/blip2_err
rm -rf ~/.cache/huggingface/hub/*

echo "done blip2"

# Evaluation
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "start evaluation"

# Qwen VL
python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True > results/qwen_3_eval
python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True > results/qwen_7_eval 2> results/qwen_7_eval_err

# Gemma
python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True > results/gemma_4_eval 2> results/gemma_4_eval_err
python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True > results/gemma_12_eval 2> results/gemma_12_eval_err

# Blip2
python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True > results/blip2_eval 2> results/blip2_eval_err

echo "done evaluation"
