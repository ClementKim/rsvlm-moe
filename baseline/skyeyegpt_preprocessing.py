import os
import json

def main():
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

if __name__ == "__main__":
    main()