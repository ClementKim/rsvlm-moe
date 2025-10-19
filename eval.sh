#!/bin/bash

source .venv/bin/activate

echo "start evaluation"

# python3 baseline/baseline_main.py --model qwen --param 3 --batch 16 --seed 42 --eval True > results/qwen_3_eval
python3 baseline/baseline_main.py --model qwen --param 7 --batch 16 --seed 42 --eval True > results/qwen_7_eval 2> results/qwen_7_eval_err

python3 baseline/baseline_main.py --model gemma --param 4 --batch 16 --seed 42 --eval True > results/gemma_4_eval 2> results/gemma_4_eval_err
python3 baseline/baseline_main.py --model gemma --param 12 --batch 16 --seed 42 --eval True > results/gemma_12_eval 2> results/gemma_12_eval_err

python3 baseline/baseline_main.py --model blip2 --param 0 --batch 16 --seed 42 --eval True > results/blip2_eval 2> results/blip2_eval_err

echo "done evaluation"
