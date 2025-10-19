import json
import torch
import argparse

from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class RS_dataset(Dataset):
    def __init__(self, dataset_dict, resolution):
        self.resolution = resolution
        self.dataset = list(dataset_dict[self.resolution].values())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        question = "[vqa]" + data['question']
        answer = data['answer']
        img = Image.open(data['image_path']).convert('RGB')

        return question, answer, img
    
def sky_collate_fn(batch):
    questions, answers, images = [], [], []
    for q, a, img in batch:
        questions.append(q)
        answers.append(a)
        images.append(img)

    
    
def main(args):
    model = torch.load("baseline/SkyEyeGPT.pth")
    model.eval()

    base_dir = "dataset/json/rsvqa"
    with open(f"{base_dir}/test.json", "r") as f:
        test = json.load(f)

    test_low_dataset = RS_dataset(test, "low")
    test_high_dataset = RS_dataset(test, "high")

    low_loader = DataLoader(
        test_low_dataset,
        batch_size = args.batch,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = True,
        prefetch_factor = 4,
        collate_fn = partial(set_collate_fn)
    )

    high_loader = DataLoader(
        test_high_dataset,
        batch_size = args.batch,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = True,
        prefetch_factor = 4,
        collate_fn = partial(set_collate_fn)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type = int, default = 16)
    args = parser.parse_args()

    main(args)