import os
import json

def rsvqa_all(dataset_dict):
    # dataset dictionary preparation
    dataset_dict['low'] = {}
    dataset_dict['high'] = {}

    for resolution in ["high", "low"]:
        for file_type in ["questions", "answers"]:
            if resolution == "high":
                target_file = "USGS"
                image_path = os.path.join(".", "rsvqa", f"{resolution}_resolution", "Images", "Data")

            elif resolution == "low":
                target_file = "all_"
                image_path = os.path.join(".", "rsvqa", f"{resolution}_resolution", "Images_LR", "Images_LR")

            with open(os.path.join(".", "rsvqa", f"{resolution}_resolution", f"{target_file}{file_type}.json"), 'r') as f:
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