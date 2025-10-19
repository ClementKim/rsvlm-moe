import argparse, torch, json, os, random
import numpy as np

from os.path import isfile
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# local modules
import baseline_models
import preprocessing
import evaluation

# qwen vision-language utils
from qwen_vl_utils import process_vision_info

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

class pretrained_model:
    def __init__(self, model, param):
        self.answer_dict = {
            "qwen": {
                "low": [],
                "high": []
                },
            "llama": {
                "low": [],
                "high": []
                },
            "gemma": {
                "low": [],
                "high": []
                },
            "blip2": {
                "low": [],
                "high": []
                }
            }

        self.model = model
        self.param = param

        if self.model == "qwen":
            self.model, self.processor = baseline_models.qwen_vl(param)

        elif self.model == "llama":
            self.model, self.processor = baseline_models.llama_vision(param)

        elif self.model == "gemma":
            self.model, self.processor = baseline_models.gemma(param)

        elif self.model == "blip2":
            self.model, self.processor = baseline_models.blip2()

def qwen_collate_fn(batch, processor):
    """
    Qwen2.5-VL-72B-Instruct 전용.
    messages -> apply_chat_template(tokenize=False) -> process_vision_info -> processor(...)
    반환: (inputs, questions, answers)
    """
    questions, answers, pil_images = [], [], []
    for q, a, img in batch:
        questions.append(q)
        answers.append(a)
        pil_images.append(img)

    # Qwen 메시지 포맷 구성
    messages_list = []
    for pil, q in zip(pil_images, questions):
        user_text = f"{q}\nYou must give fianl answer in one sentence"
        messages_list.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text",  "text": user_text},
                ],
            }
        ])

    # 텍스트 템플릿 및 비전 입력 준비
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]
    image_inputs, video_inputs = process_vision_info(messages_list)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs, questions, answers


def llama_collate_fn(batch, processor):
    """
    Llama-3.2-90B-Vision-Instruct 전용.
    messages -> apply_chat_template(tokenize=False) -> processor(images=..., text=...)
    """
    questions, answers, pil_images = [], [], []
    for q, a, img in batch:
        questions.append(q)
        answers.append(a)
        pil_images.append([img])

    texts = []
    for q in questions:
        user_text = f"{q} You must give fianl answer in one sentence."
        messages = [
            {"role": "user",
             "content": [
                 {"type": "image"},                 # 실제 이미지는 images 인자로 전달
                 {"type": "text", "text": user_text},
             ]}
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    inputs = processor(
        images=pil_images,
        text=texts,
        padding=True,
        return_tensors="pt",
    )
    return inputs, questions, answers


def gemma_collate_fn(batch, processor):
    """
    gemma-3-27b-it 전용 (멀티모달).
    apply_chat_template을 사용하여 정확한 프롬프트 형식을 생성합니다.
    """
    questions, answers, pil_images = [], [], []
    for q, a, img in batch:
        questions.append(q)
        answers.append(a)
        pil_images.append([img])

    texts = []
    for q in questions:
        # CORRECT: Structure content as a list of dictionaries
        user_text = f"{q} You must give fianl answer in one sentence."
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ]
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    inputs = processor(
        text=texts,
        images=pil_images,
        padding=True,
        return_tensors="pt",
    )

    return inputs, questions, answers

def blip2_collate_fn(batch, processor):
    questions, answers, pil_images = [], [], []
    for q, a, img in batch:
        questions.append(q)
        answers.append(a)
        pil_images.append(img)
    
    texts = []
    for q in questions:
        texts.append(f"Question: {q} You must give fianl answer in one sentence.\nAnswer:")

    inputs = processor(
        images = pil_images,
        text = texts,
        padding = True,
        return_tensors = "pt"
    )

    return inputs, questions, answers

def main(args):
    # control randomness
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if (args.param == 0):
        args.param = None

    if (args.eval == "True"):
        args.eval = True

    else:
        args.eval = False

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

    if not (args.eval):
        # 모델 로드
        vlm = pretrained_model(args.model, args.param)
        vlm.model.eval()

        if args.model == "qwen":
            set_collate_fn = qwen_collate_fn

        elif args.model == "llama":
            set_collate_fn = llama_collate_fn

        elif args.model == "gemma":
            set_collate_fn = gemma_collate_fn

        elif args.model == "blip2":
            set_collate_fn = blip2_collate_fn

        low_loader = DataLoader(
            test_low_dataset,
            batch_size = args.batch,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
            prefetch_factor = 4,
            collate_fn = partial(set_collate_fn, processor = vlm.processor)
        )

        high_loader = DataLoader(
            test_high_dataset,
            batch_size = args.batch,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
            prefetch_factor = 4,
            collate_fn = partial(set_collate_fn, processor = vlm.processor)
        )

        for batch_inputs, questions, answers in tqdm(low_loader, desc = f"{args.model} Inference (Low Resolution)"):
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
                vlm.answer_dict[args.model]["high"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens = True)[0].strip())
            else:
                vlm.answer_dict[args.model]["high"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens=True))
        
        # 결과 저장 (low 우선 저장)
        os.makedirs("./results", exist_ok=True)
        with open(f"./results/{args.model}_{args.param}_results.json", "w") as f:
            json.dump(vlm.answer_dict[args.model], f, indent=4)
        
        for batch_inputs, questions, answers in tqdm(high_loader, desc = f"{args.model} Inference (High Resolution)"):
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
                vlm.answer_dict[args.model]["high"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens = True)[0].strip())
            else:
                vlm.answer_dict[args.model]["high"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens=True))

        # 결과 저장 (high 포함)
        with open(f"./results/{args.model}_{args.param}_results.json", "w") as f:
            json.dump(vlm.answer_dict[args.model], f, indent=4)

    else:
        low_answer, high_answer = evaluation.ExtractResponse(args.model, str(args.param))
        low_reference, high_reference = evaluation.ExtractReference(test)

        # Bert Score
        precision_low, recall_low, f1_low = evaluation.bert_score(low_reference, low_answer, device)
        precision_high, recall_high, f1_high = evaluation.bert_score(high_reference, high_answer, device)

        # Bleu score
        bleu_low = evaluation.bleu_score(low_reference, low_answer)
        bleu_high = evaluation.bleu_score(high_reference, high_answer)

        # Rouge score
        rouge_low = evaluation.rouge_score(low_reference, low_answer)
        rouge_high = evaluation.rouge_score(high_reference, high_answer)

        # Meteor score
        meteor_low = evaluation.meteor_score(low_reference, low_answer)
        meteor_high = evaluation.meteor_score(high_reference, high_answer)

        # print results
        print(f"{args.model} [Low Resolution] BERTScore - Precision: {precision_low:.4f}, Recall: {recall_low:.4f}, F1: {f1_low:.4f}")
        print(f"{args.model} [Low Resolution] BLEU Score: {bleu_low:.4f}")
        print(f"{args.model} [Low Resolution] ROUGE Score: {rouge_low}")
        # print(f"{args.model} [Low Resolution] METEOR Score: {meteor_low}")

        print(f"{args.model} [High Resolution] BERTScore - Precision: {precision_high:.4f}, Recall: {recall_high:.4f}, F1: {f1_high:.4f}")
        print(f"{args.model} [High Resolution] BLEU Score: {bleu_high:.4f}")
        print(f"{args.model} [High Resolution] ROUGE Score: {rouge_high}")
        # print(f"{args.model} [High Resolution] METEOR Score: {meteor_high}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "qwen")
    parser.add_argument("--param", type = int, default = 3)
    parser.add_argument("--batch", type = int, default = 16)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--eval", type = str, default = "True")
    args = parser.parse_args()

    main(args)
