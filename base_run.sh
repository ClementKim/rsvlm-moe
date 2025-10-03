#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH=2
SEED=42

source .venv/bin/activate

python3 baseline/baseline_main.py --model qwen --batch $BATCH --seed $SEED 2> results/qwen_err

rm -rf ~/.cache/huggingface/hub/*

python3 baseline/baseline_main.py --model llama --batch $BATCH --seed $SEED 2> results/llama_err

rm -rf ~/.cache/huggingface/hub/*

python3 baseline/baseline_main.py --model gemma --batch $BATCH --seed $SEED 2> results/gemma_err

rm -rf ~/.cache/huggingface/hub/*
