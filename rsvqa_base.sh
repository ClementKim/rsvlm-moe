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
python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/qwen_3_err
rm -rf ~/.cache/huggingface/hub/*

# Qwen VL 7B
python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/qwen_7_err
rm -rf ~/.cache/huggingface/hub/*

echo "done Qwen"

# Llama
# echo "start llama"

# python3 baseline/baseline_main.py --model llama --param 11 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/llama_11_err
# rm -rf ~/.cache/huggingface/hub/*

# python3 baseline/baseline_main.py --model llama --param 90 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/llama_90_err
# rm -rf ~/.cache/huggingface/hub/*

# echo "done llama"

# Gemma
echo "start gemma"

# Gemma 4B
python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/gemma_4_err
rm -rf ~/.cache/huggingface/hub/*

# Gemma 12B
python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/gemma_12_err
rm -rf ~/.cache/huggingface/hub/*

# Gemma 27B
# python3 baseline/baseline_main.py --model gemma --param 27 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/gemma_27_err
# rm -rf ~/.cache/huggingface/hub/*

echo "done gemma"

# Blip2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "start blip2"

python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval False 2> results/@2/blip2_err
rm -rf ~/.cache/huggingface/hub/*

echo "done blip2"

# GeoChat
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

deactivate
conda activate geochat

# python3 baseline/geochat_preprocessing.py

cd baseline/GeoChat

echo "start geochat"
python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_low.json --answers-file ~/js_data/rsvlm/results/@2/geochat-low.json 2> ~/js_data/rsvlm/results/@2/geochat_low_err
python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_high.json --answers-file ~/js_data/rsvlm/results/@2/geochat-high.json 2> ~/js_data/rsvlm/results/@2/geochat_high_err

cd ../..

conda deactivate

# SkyEyeGPT
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=1

conda activate minigptv

export port=2702
export cfg_path="eval_configs/minigptv2_eval_low.yaml"

echo "start SkyEyeGPT"
cd baseline/MiniGPT-4
torchrun --master-port ${port} --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ${cfg_path} --dataset rsvqa 2> ~/js_data/rsvlm/results/@2/sky_low_err

export cfg_path="eval_configs/minigptv2_eval_high.yaml"
torchrun --master-port ${port} --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ${cfg_path} --dataset rsvqa 2> ~/js_data/rsvlm/results/@2/sky_high_err

conda deactivate

cd ~/js_data/rsvlm

# Evaluation
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "start evaluation"
source .venv/bin/activate

echo "qwen 3"
python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/@2/qwen_3_eval 2> results/@2/qwen_3_eval_err

echo "qwen 7"
python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/@2/qwen_7_eval 2> results/@2/qwen_7_eval_err

echo "gemma 4"
python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/@2/gemma_4_eval 2> results/@2/gemma_4_eval_err

echo "gemma 12"
python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/@2/gemma_12_eval 2> results/@2/gemma_12_eval_err

echo "blip2"
python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/@2/blip2_eval 2> results/@2/blip2_eval_err

echo "geochat"
python3 baseline/baseline_main.py --model geochat --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/@2/geochat_eval 2> results/@2/geochat_eval_err

echo "skyeyegpt"
python3 baseline/baseline_main.py --model skyeyegpt --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --eval True --eval True > results/@2/skyeyegpt_eval 2> results/@2/skyeyegpt_eval_err

echo "done evaluation"
