import torch
import io
import os
import json
import argparse

from evaluation import evaluation_main

from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from google import generativeai as genai
from google.generativeai import types

def prepare_dataset():
    # 데이터 준비
    test_dir = "dataset/json/rsvqa/test.json"
    with open(test_dir, 'r') as f:
        test = json.load(f)

    return test

def gemini_test(api, param, question, img):
    genai.configure(api_key=api)
    model_name = f'gemini-2.5-{param}'
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([question, img])

    return response.text

# def gemini_test(api,question, img):
#     return 1

def openai_test(api, question, img):
    return 1

def str_to_bool(s):
    if s.lower() == 'true':
        return True
    
    elif s.lower() == 'false':
        return False
    
    else:
        raise ValueError("Boolean value expected.")

def main(args):
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API")
    openai_api_key = os.getenv("OPENAI_API")

    test_dataset = prepare_dataset()

    if str_to_bool(args.evaluation):
        evaluation_main(param = args.param, device = "cuda" if torch.cuda.is_available() else "cpu", test = test_dataset)
        return

    gemini_response = {"low": {},
                       "high": {}}
    
    # openai_response = {"low": {},
    #                    "high": {}}
    for resol, value in test_dataset.items():
        for question_id, question_set in tqdm(value.items()):
            question = f"{question_set['question']}. Answer in one sentence."
            img_path = question_set['image_path']

            img = Image.open(img_path)

            img_byte_arr = io.BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()

            image_part = {
                'mime_type': 'image/jpeg',
                'data': img_bytes
            }

            gemini_response[resol][question_id] = gemini_test(gemini_api_key, args.param, question, image_part)
            # openai_response[resol][question_id] = openai_test(openai_api_key, question, img)

        break

    with open(f"results/gemini_{args.param}_results.json", "w") as f:
        json.dump(gemini_response, f, indent = 4)

    # with open("results/openai_results.json", "w") as f:
    #     json.dump(openai_response, f, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type = str, default = "False")
    parser.add_argument("--param", type = str, default = "flash")
    args = parser.parse_args()
    main(args)