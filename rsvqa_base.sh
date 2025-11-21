#!/bin/bash

# Set CUDA devices and PyTorch configuration
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
source .venv/bin/activate # venv environment
# conda activate rsvlm # conda environment

# Set batch size and seed
for SEED in 42 43 44 777 911; do
for PROMPT in "False" "True"; do
BATCH=32

if [ "$SEED" -eq 42 ] && [ "$PROMPT" == False ]; then
    continue
fi

# Qwen
echo "start Qwen"

# Qwen VL 3B
python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval False 2> results/qwen_3_$SEED_prompt_$PROMPT_err
rm -rf ~/.cache/huggingface/hub/*

# Qwen VL 7B
python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval False 2> results/qwen_7_$SEED_prompt_$PROMPT_err
rm -rf ~/.cache/huggingface/hub/*

echo "done Qwen"

# Llama
# echo "start llama"

# python3 baseline/baseline_main.py --model llama --param 11 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval False 2> results/llama_11_$SEED_prompt_$PROMPT_err
# rm -rf ~/.cache/huggingface/hub/*

# echo "done llama"

# Gemma
echo "start gemma"

# Gemma 4B
python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval False 2> results/gemma_4_$SEED_prompt_$PROMPT_err
rm -rf ~/.cache/huggingface/hub/*

# Gemma 12B
python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval False 2> results/gemma_12_$SEED_prompt_$PROMPT_err
rm -rf ~/.cache/huggingface/hub/*

echo "done gemma"

# Blip2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "start blip2"

python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval False 2> results/blip2_$SEED_prompt_$PROMPT_err
rm -rf ~/.cache/huggingface/hub/*

echo "done blip2"

# GeoChat
unset PYTORCH_CUDA_ALLOC_CONF
unset CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

deactivate
conda activate geochat

# python3 baseline/geochat_preprocessing.py

cd baseline/GeoChat

echo "start geochat"
python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_low.json --answers-file ~/js_data/rsvlm/results/geochat-low-$SEED-prompt-$PROMPT.json 2> ~/js_data/rsvlm/results/geochat_low_$SEED_prompt_$PROMPT_err
python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_high.json --answers-file ~/js_data/rsvlm/results/geochat-high-$SEED-prompt-$PROMPT.json 2> ~/js_data/rsvlm/results/geochat_high_$SEED_prompt_$PROMPT_err

cd ../..

conda deactivate

# SkyEyeGPT
unset PYTORCH_CUDA_ALLOC_CONF
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1

conda activate minigptv

export port=2702
export cfg_path="eval_configs/minigptv2_eval_low.yaml"

echo "start SkyEyeGPT"
cd baseline/MiniGPT-4
torchrun --master-port ${port} --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ${cfg_path} --dataset rsvqa 2> ~/js_data/rsvlm/results/sky_low_$SEED_prompt_$PROMPT_err

export cfg_path="eval_configs/minigptv2_eval_high.yaml"
torchrun --master-port ${port} --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ${cfg_path} --dataset rsvqa 2> ~/js_data/rsvlm/results/sky_high_$SEED_prompt_$PROMPT_err

conda deactivate

cd ~/js_data/rsvlm

# Evaluation
unset PYTORCH_CUDA_ALLOC_CONF
unset CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "start evaluation"
source .venv/bin/activate

echo "qwen 3"
python3 baseline/baseline_main.py --model qwen --param 3 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval True > results/qwen_3_eval 2> results/qwen_3_eval_$SEED_prompt_$PROMPT_err

echo "qwen 7"
python3 baseline/baseline_main.py --model qwen --param 7 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval True > results/qwen_7_eval 2> results/qwen_7_eval_$SEED_prompt_$PROMPT_err

echo "gemma 4"
python3 baseline/baseline_main.py --model gemma --param 4 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval True > results/gemma_4_eval 2> results/gemma_4_eval_$SEED_prompt_$PROMPT_err

echo "gemma 12"
python3 baseline/baseline_main.py --model gemma --param 12 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval True > results/gemma_12_eval 2> results/gemma_12_eval_$SEED_prompt_$PROMPT_err

echo "blip2"
python3 baseline/baseline_main.py --model blip2 --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval True > results/blip2_eval 2> results/blip2_eval_$SEED_prompt_$PROMPT_err

echo "geochat"
python3 baseline/baseline_main.py --model geochat --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval True > results/geochat_eval 2> results/geochat_eval_$SEED_prompt_$PROMPT_err

echo "skyeyegpt"
python3 baseline/baseline_main.py --model skyeyegpt --param 0 --batch $BATCH --seed $SEED --dataset rsvqa --train False --val False --test True --prompt $PROMPT --eval True > results/skyeyegpt_eval 2> results/skyeyegpt_eval_$SEED_prompt_$PROMPT_err

echo "done evaluation"

done
done