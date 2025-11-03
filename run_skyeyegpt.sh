#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

conda activate minigptv

export port=2702
export cfg_path="eval_configs/minigptv2_eval_low.yaml"

cd baseline/MiniGPT-4
torchrun --master-port ${port} --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ${cfg_path} --dataset rsvqa 2> ~/js_data/rsvlm/results/sky_low_err

export cfg_path="eval_configs/minigptv2_eval_high.yaml"
torchrun --master-port ${port} --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ${cfg_path} --dataset rsvqa 2> ~/js_data/rsvlm/results/sky_high_err

conda deactivate

cd ~/js_data/rsvlm