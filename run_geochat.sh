#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

conda activate geochat

# python3 baseline/geochat_preprocessing.py

cd baseline/GeoChat

python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_low.json --answers-file ~/js_data/rsvlm/results/geochat-low.json 2> ~/js_data/rsvlm/results/geochat_low_err
python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_high.json --answers-file ~/js_data/rsvlm/results/geochat-high.json 2> ~/js_data/rsvlm/results/geochat_high_err

cd ../..

conda deactivate

# source .venv/bin/activate