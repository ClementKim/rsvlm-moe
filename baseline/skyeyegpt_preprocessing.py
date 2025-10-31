import os
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

if __name__ == "__main__":
    main()