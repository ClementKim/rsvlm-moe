#!/bin/bash

python3 llm/main.py --evaluation False --param flash 2> results/gemini_flash_err
python3 llm/main.py --evaluation True --param flash > results/gemini_flash_eval 2> results/gemini_flash_eval_err

python3 llm/main.py --evaluation False --param pro 2> results/gemini_pro_err
python3 llm/main.py --evaluation True --param pro > results/gemini_pro_eval 2> results/gemini_pro_eval_err
