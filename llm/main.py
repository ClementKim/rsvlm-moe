import torch
import io
import os
import json
import jsonlines
import argparse
import time

from evaluation import evaluation_main

from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types

uploaded_files_cache = {}

def prepare_dataset():
    # 데이터 준비
    test_dir = "dataset/json/rsvqa/test.json"
    with open(test_dir, 'r') as f:
        test = json.load(f)

    return test

def str_to_bool(s):
    if s.lower() == 'true':
        return True
    
    elif s.lower() == 'false':
        return False
    
    else:
        raise ValueError("Boolean value expected.")
    
def upload_image(client, img_path):
    if img_path in uploaded_files_cache:
        return uploaded_files_cache[img_path]
    
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    temp_filename = ''.join(os.path.basename(img_path).rsplit('.', 1)[:1]) + '_temp.jpg'
    img.save(temp_filename, format='JPEG', quality=90)

    uploaded_file = client.files.upload(
        file = temp_filename,
        config = types.UploadFileConfig(
            mime_type = 'image/jpeg',
            display_name = os.path.basename(temp_filename)
        ))

    while uploaded_file.state.name == 'PROCESSING':
        time.sleep(1)
        uploaded_file = client.files.get(uploaded_file.name)
    uploaded_files_cache[img_path] = uploaded_file.uri

    return uploaded_file.uri

def main(args):
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API")
    client = genai.Client(api_key=gemini_api_key)
    # genai.configure(api_key=gemini_api_key)

    test_dataset = prepare_dataset()

    if str_to_bool(args.evaluation):
        evaluation_main(param = args.param, device = "cuda" if torch.cuda.is_available() else "cpu", test = test_dataset)
        return

    args.prompt = str_to_bool(args.prompt)

    batch_input_filename = f"batch_input_{args.param}.jsonl"
    
    prompt = """# Identity
                You are an expert in remote sensing image understanding and analysis, especially for visual question answering about aerial image tasks.
                
                # Instructions
                * Perspective Awareness: Always analyze images assuming a 'nadir' (bird's-eye) or high angle aerial view. Don't interpret flat roofs as floors or roads as walls.
                * Counting precision: When asked to count objects (e.g., buildings, vehicles, planes), be extremely rigorous. If objects are clustered, estimate based on density but prioritize individual distinct features.
                * Consisness: Provide direct, factual answers. Don't add 'I think' or 'it appears to be'
                * Format: Return only the final answer string (e.g., 'yes', '5', 'residential area') without markdown formatting or conversational filler, unless the user explicitly asks for an explanation.
                
                # Examples
                <User_Query>
                [Image Context: Aerial view of a suburban neighborhood]
                Is this area urban or rural?
                </User_Query>
                <Assistant_Response>
                urban
                </Assistant_Response>
                
                <User_Query>
                [Image Context: Aerial view of a Wall-Mart parking lot filled with cars]
                How many vehicles are visible in the parking lot?
                </User_Query>
                <Assistant_Response>
                126
                </Assistant_Response>
                
                # User Query"""
    
    with open(batch_input_filename, 'w', encoding = "utf-8") as f:
        for resol, value in test_dataset.items():
            response_num = 1
            for question_id, question_set in tqdm(value.items()):
                if not args.prompt:
                    question = f"{question_set['question']}. Answer in one sentence."

                else:
                    question = f"{prompt}\n{question_set['question']}"

                img_path = question_set['image_path']

                try:
                    file_uri = upload_image(client, img_path)

                except Exception as e:
                    print(f"Image upload failed for {img_path}: {e}")
                    continue

                custom_id = f"{resol}|{question_id}"
                temp = {f"request": {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [
                                        {"text": question},
                                        {"file_data": {"mime_type": "image/jpeg", "file_uri": file_uri}}
                                    ]
                                }
                            ]
                        },
                        "key": custom_id
                    }
                
                response_num += 1
                
                f.write(json.dumps(temp) + "\n")


    batch_input_file = client.files.upload(
        file = batch_input_filename,
        config = types.UploadFileConfig(
            display_name = "RSVQA Batch Input File",
            mime_type = "application/jsonl"
        ))

    model_name = f"models/gemini-2.5-{args.param}"

    try:
        batch_job = client.batches.create(
            src = batch_input_file.name,
            model = model_name,
            config = {
                "display_name": f"rsvqa_batch_{args.param}",
            }
        )

    except Exception as e:
        print(f"Batch job creation failed: {e}")
        return
    
    job_name = batch_job.name
    batch_job = client.batches.get(name = job_name)

    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    
    while batch_job.state.name not in completed_states:
        time.sleep(30)
        batch_job = client.batches.get(name = job_name)

    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Batch job failed to start: {batch_job.state.message}")
        return
    
    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name
            file_content = client.files.download(file=result_file_name)
            with open("batch_job_results.jsonl", "wb") as f:
                f.write(file_content)
                
        elif batch_job.dest and batch_job.dest.inlined_responses:
            print("Results are inline, not in file.")

        else:
            print("No results found (neither file nor inline).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type = str, default = "False")
    parser.add_argument("--param", type = str, default = "flash")
    parser.add_argument("--prompt", type = str, default = "False")
    args = parser.parse_args()
    main(args)