#!/bin/bash

source .venv/bin/activate

echo "start evaluation"

BATCH=32
SEED=42

echo "qwen 3"
python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/qwen_3_eval 2> results/qwen_3_eval_err

echo "qwen 7"
python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/qwen_7_eval 2> results/qwen_7_eval_err

echo "gemma 4"
python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/gemma_4_eval 2> results/gemma_4_eval_err

echo "gemma 12"
python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/gemma_12_eval 2> results/gemma_12_eval_err

echo "blip2"
python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/blip2_eval 2> results/blip2_eval_err

echo "geochat"
python3 baseline/baseline_main.py --model geochat --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/geochat_eval 2> results/geochat_eval_err

echo "skyeyegpt"
python3 baseline/baseline_main.py --model skyeyegpt --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/skyeyegpt_eval 2> results/skyeyegpt_eval_err

echo "done evaluation"
