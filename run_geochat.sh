#!/bin/bash

conda activate geochat
cd baseline/GeoChat

python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_low.json --answer-file results/geochat-low.json
python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file ~/js_data/rsvlm/dataset/json/rsvqa/test_high.json --answer-file results/geochat-high.json

cd ../..

conda deactivate

source .venv/bin/activate