#!/bin/bash

conda activate geochat

python3 baseline/geochat_preprocessing.py

cd baseline/GeoChat

python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file /home/jovyan/js_data/rsvlm/dataset/json/rsvqa/test_low.json --answers-file /mnt/d/eccv26/results/geochat-low.json 2> ~/js_data/rsvlm/results/geochat_low
python3 geochat/eval/batch_geochat_vqa.py --model-path MBZUAI/geochat-7B --question-file /home/jovyan/js_data/rsvlm/dataset/json/rsvqa/test_high.json --answers-file /mnt/d/eccv26/results/geochat-high.json 2> ~/js_data/rsvlm/results/geochat_high

cd ../..

conda deactivate

# source .venv/bin/activate