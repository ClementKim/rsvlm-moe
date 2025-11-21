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
    
def loader(args, dataset, collate_fn, processor, prompt_str):
    return DataLoader(
        dataset,
        batch_size = args.batch,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = True,
        prefetch_factor = 4,
        collate_fn = partial(collate_fn, processor = processor, prompt = args.prompt, prompt_str = prompt_str)
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

    if args.train:
        return train_low_dataset, train_high_dataset, train
    elif args.val:
        return val_low_dataset, val_high_dataset, val
    elif args.test:
        return test_low_dataset, test_high_dataset, test
    
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

    # prompt
    prompt = """
                # Identity
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
                
                # User Query
                
        """

    if args.eval:
        evaluation_main(args, device, full_dataset)

    elif not (args.eval):
        answer_dict = {
            "low": [],
            "high": []
        }
        
        for dataset in [low_dataset, high_dataset]:
            data_loader = loader(args, dataset, vlm.collate_fn, vlm.processor, prompt)

            if dataset == low_dataset:
                data_type = "low"
            else:
                data_type = "high"

            for batch_inputs, questions, answers in tqdm(data_loader, desc = f"{args.model} Inference ({dataset})"):
                batch_inputs = {k: v.to(vlm.model.device) if hasattr(v, "to") else v
                                for k, v in batch_inputs.items()}
                with torch.inference_mode():
                    if args.model == "blip2":
                        input_len = batch_inputs["input_ids"].shape[-1] if "input_ids" in batch_inputs else 0
                        max_pos = getattr(vlm.model.config, "max_position_embeddings", 2048)
                        headroom = max_pos - input_len
                        safe_new_tokens = max(1, min(64, headroom - 1))

                        gen_ids = vlm.model.generate(
                            **batch_inputs,
                            max_new_tokens=safe_new_tokens,
                            eos_token_id=vlm.processor.tokenizer.eos_token_id,
                            pad_token_id=vlm.processor.tokenizer.pad_token_id,
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
                    answer_dict[data_type].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens = True)[0].strip())
                else:
                    answer_dict[data_type].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens=True))
            
            os.makedirs("./results", exist_ok=True)
            with open(f"./results/{args.model}_{args.param}_{args.seed}_prompt_{args.prompt}_results.json", "w") as f:
                json.dump(answer_dict, f, indent=4)