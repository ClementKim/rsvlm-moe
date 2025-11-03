import os
import json

from random import shuffle

def rsvqa_all(dataset_dict):
    # dataset dictionary preparation
    dataset_dict['low'] = {}
    dataset_dict['high'] = {}

    for resolution in ["high", "low"]:
        for file_type in ["questions", "answers"]:
            if resolution == "high":
                target_file = "USGS"
                image_path = os.path.join(".", "dataset", "rsvqa", f"{resolution}_resolution", "images")

            elif resolution == "low":
                target_file = "all_"
                image_path = os.path.join(".", "dataset", "rsvqa", f"{resolution}_resolution", "Images_LR")

            with open(os.path.join(".", "dataset", "rsvqa", f"{resolution}_resolution", f"{target_file}{file_type}.json"), 'r') as f:
                json_data = json.load(f)

                if file_type == "questions":
                    for key, lst in json_data.items():
                        for item in lst:
                            dataset_dict[resolution][item['id']] = {'question': item['question'],
                                                                    'image_path': os.path.join(image_path, f"{item['img_id']}.tif"),
                                                                    'answer': 0}

                elif file_type == "answers":
                    for key, lst in json_data.items():
                        for item in lst:
                            dataset_dict[resolution][item['question_id']]['answer'] = item['answer']

    with open("dataset/json/rsvqa/dataset_dict.json", 'w') as f:
        json.dump(dataset_dict, f)

    return dataset_dict

def rsvqa_split(dataset_dict):
    train, val, test = {}, {}, {}

    low_ids = []
    for question_id, _ in dataset_dict["low"].items():
        low_ids.append(question_id)

    high_ids = []
    for question_id, _ in dataset_dict["high"].items():
        high_ids.append(question_id)

    train['low'] = {}
    train['high'] = {}

    val['low'] = {}
    val['high'] = {}

    test['low'] = {}
    test['high'] = {}

    shuffle(low_ids)
    shuffle(high_ids)

    train_rate, val_rate = len(low_ids) * 0.6, len(low_ids) * 0.3
    for idx, num in enumerate(low_ids, start = 1):
        if idx <= train_rate:
            train["low"][num] = dataset_dict['low'][num]

        elif idx <= train_rate + val_rate:
            val["low"][num] = dataset_dict['low'][num]

        elif idx > train_rate + val_rate:
            test["low"][num] = dataset_dict['low'][num]

    train_rate, val_rate = len(high_ids) * 0.6, len(high_ids) * 0.3
    for idx, num in enumerate(high_ids, start = 1):
        if idx <= train_rate:
            train["high"][num] = dataset_dict['high'][num]

        elif idx <= train_rate + val_rate:
            val["high"][num] = dataset_dict['high'][num]

        elif idx > train_rate + val_rate:
            test["high"][num] = dataset_dict['high'][num]

    save_dir = "dataset/json/rsvqa"
    with open(f"{save_dir}/train.json", "w") as f:
        json.dump(train, f, indent = 4)

    with open(f"{save_dir}/val.json", "w") as f:
        json.dump(val, f, indent = 4)

    with open(f"{save_dir}/test.json", "w") as f:
        json.dump(test, f, indent = 4)

    return train, val, test

def rsvqa_geochat():
    base_dir = "dataset/json/rsvqa"
    for ty in ["train", "val", "test"]:
        low_temp_list = []
        high_temp_list = []
        file = f"{base_dir}/{ty}.json"
        with open(file, "r") as f:
            temp_dict = json.load(f)

        for resol, value in temp_dict.items():
            for id, qa in value.items():
                image_path = "/home/jovyan/js_data/rsvlm" + qa['image_path'][1:]

                if not os.path.exists(image_path):
                    image_path = "/mnt/d/eccv26" + qa['image_path'][1:]

                item = {
                    "question_id": f"rsvqa-{resol}-{id}",
                    "image": image_path,
                    "text": f"\n{qa['question']} You must give final answer in one sentence."
                }

                if resol == "low":
                    low_temp_list.append(item)

                else:
                    high_temp_list.append(item)

        with open(f"{base_dir}/{ty}_low.json", "w") as f:
            for item in low_temp_list:
                f.write(json.dumps(item) + "\n")

        with open(f"{base_dir}/{ty}_high.json", "w") as f:
            for item in high_temp_list:
                f.write(json.dumps(item) + "\n")

def rsvqa_skyeyegpt():
    base_dir = "dataset/json/rsvqa"
    save_dir = "baseline/MiniGPT-4/dataset/rsvqa"

    for ty in ["train", "val", "test"]:
        low_temp_list = []
        high_temp_list = []
        file = f"{base_dir}/{ty}.json"
        with open(file, "r") as f:
            temp_dict = json.load(f)

        for resol, value in temp_dict.items():
            for id, qa in value.items():
                image_id = qa['image_path'].split('/')[-1]

                item = {
                    "question": f"\n{qa['question']} You must give final answer in one sentence.",
                    "image": image_id,
                    "question_id": f"rsvqa-{resol}-{id}",
                }

                if resol == "low":
                    low_temp_list.append(item)

                else:
                    high_temp_list.append(item)

        with open(f"{save_dir}/{ty}_low.json", "w") as f:
            for item in low_temp_list:
                f.write(json.dumps(item) + "\n")

        with open(f"{save_dir}/{ty}_high.json", "w") as f:
            for item in high_temp_list:
                f.write(json.dumps(item) + "\n")