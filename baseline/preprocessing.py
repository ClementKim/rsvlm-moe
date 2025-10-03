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

    train_rate, val_rate = len(low_ids) * 0.6, len(low_ids) * 0.2
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

    return train, val, test