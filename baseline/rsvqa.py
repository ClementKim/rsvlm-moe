import os
import json
import torch
import random
import numpy as np
import preprocessing

from tqdm import tqdm
from PIL import Image
from os.path import isfile
from evaluation import evaluation_main
from torch.utils.data import Dataset, DataLoader
from functools import partial

class RS_dataset(Dataset):
    def __init__(self, dataset_dict, resolution):
        self.resolution = resolution
        self.dataset = list(dataset_dict[self.resolution].values())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        question = data['question']
        answer = data['answer']
        img = Image.open(data['image_path']).convert('RGB')

        return question, answer, img
    
def loader(args, dataset, collate_fn, processor):
    return DataLoader(
        dataset,
        batch_size = args.batch,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = True,
        prefetch_factor = 4,
        collate_fn = partial(collate_fn, processor = processor)
    )
    
def rsvqa_dataset(args):
    # 데이터 준비
    target_dir = "dataset/json/rsvqa/dataset_dict.json"
    train_dir = "dataset/json/rsvqa/train.json"
    val_dir = "dataset/json/rsvqa/val.json"
    test_dir = "dataset/json/rsvqa/test.json"
    if not (isfile(target_dir)):
        dataset_dict = preprocessing.rsvqa_all({})
        train, val, test = preprocessing.rsvqa_split(dataset_dict)

    elif not (isfile(train_dir) and isfile(val_dir) and isfile(test_dir)):
        with open(target_dir, 'r') as f:
            dataset_dict = json.load(f)

        train, val, test = preprocessing.rsvqa_split(dataset_dict)

    else:
        with open(train_dir, "r") as f:
            train = json.load(f)

        with open(val_dir, 'r') as f:
            val = json.load(f)

        with open(test_dir, 'r') as f:
            test = json.load(f)

    train_low_dataset = RS_dataset(train, "low")
    val_low_dataset = RS_dataset(val, "low")
    test_low_dataset = RS_dataset(test, "low")

    train_high_dataset = RS_dataset(train, "high")
    val_high_dataset = RS_dataset(val, "high")
    test_high_dataset = RS_dataset(test, "high")

    return train_low_dataset, train_high_dataset, train if args.train else val_low_dataset, val_high_dataset, val if val else test_low_dataset, test_high_dataset, test

def rsvqa_main(args, vlm, device):
    # control randomness
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load datasets
    low_dataset, high_dataset, full_dataset = rsvqa_dataset(args)

    if args.eval:
        evaluation_main(args, device, full_dataset)

    elif not (args.eval):
        answer_dict = {
            "low": [],
            "high": []
        }
        
        for dataset in [low_dataset, high_dataset]:
            data_loader = loader(args, dataset, vlm.collate_fn, vlm.processor)

            for batch_inputs, questions, answers in tqdm(data_loader, desc = f"{args.model} Inference (Low Resolution)"):
                batch_inputs = {k: v.to(vlm.model.device) if hasattr(v, "to") else v
                                for k, v in batch_inputs.items()}
                with torch.inference_mode():
                    if args.model == "blip2":
                        gen_ids = vlm.model.generate(
                            **batch_inputs,
                            do_sample = True,
                            max_new_tokens = 4096,
                            min_new_tokens = 10,
                            temperature = 1.0
                        )

                    elif args.model == "instructblip":
                        gen_ids = vlm.model.generate(
                            **batch_inputs,
                            do_sample = False,
                            num_beams = 5,
                            max_length = 256,
                            min_length = 10,
                            top_p = 0.9,
                            repetition_penalty = 1.5,
                            length_penalty = 1.0,
                            temperature = 1.0
                        )
                        
                    else:
                        gen_ids = vlm.model.generate(
                            **batch_inputs,
                            do_sample = False,
                            max_new_tokens=4096,
                            min_new_tokens=10,
                            temperature=1.0)
                        
                if args.model == "instructblip":
                    answer_dict["low"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens = True)[0].strip())
                else:
                    answer_dict["low"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens=True))
            
            os.makedirs("./results", exist_ok=True)
            with open(f"./results/{args.model}_{args.param}_results.json", "w") as f:
                json.dump(answer_dict[args.model], f, indent=4)