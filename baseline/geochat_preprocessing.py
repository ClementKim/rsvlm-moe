import json

def main():
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
                item = {
                    "id": f"rsvqa-{resol}-{id}",
                    "image": image_path,
                    "conversation": [
                        {
                            "from": "human",
                            "value": f"<image>\n{qa['question']} You must give fianl answer in one sentence."
                        },
                        {
                            "from": "gpt",
                            "value": qa['answer']
                        }
                    ]
                }

                if resol == "low":
                    low_temp_list.append(item)

                else:
                    high_temp_list.append(item)

        with open(f"{base_dir}/{ty}_low.json", "w") as f:
            json.dump(low_temp_list, f, indent = 4)

        with open(f"{base_dir}/{ty}_high.json", "w") as f:
            json.dump(high_temp_list, f, indent = 4)

if __name__ == "__main__":
    main()