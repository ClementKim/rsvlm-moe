import argparse, torch, json, os, random
import numpy as np

from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# local modules
import baseline_models
import preprocessing

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
    def __init__(self, model):
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
                }
            }

        self.model = model

        if self.model == "qwen":
            self.model, self.processor = baseline_models.qwen_vl()

        elif self.model == "llama":
            self.model, self.processor = baseline_models.llama_vision()

        elif self.model == "gemma":
            self.model, self.processor = baseline_models.gemma()

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
        messages_list.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text",  "text": q},
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
        pil_images.append(img)

    texts = []
    for q in questions:
        messages = [
            {"role": "user",
             "content": [
                 {"type": "image"},                 # 실제 이미지는 images 인자로 전달
                 {"type": "text", "text": q},
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
        pil_images.append(img)

    texts = []
    for q in questions:
        # CORRECT: Structure content as a list of dictionaries
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
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

    # 데이터 준비
    dataset_dict = preprocessing.rsvqa_all({})
    train, val, test = preprocessing.rsvqa_split(dataset_dict)

    train_low_dataset = RS_dataset(train, "low")
    val_low_dataset = RS_dataset(val, "low")
    test_low_dataset = RS_dataset(test, "low")

    train_high_dataset = RS_dataset(train, "high")
    val_high_dataset = RS_dataset(val, "high")
    test_high_dataset = RS_dataset(test, "high")

    # 모델 로드
    vlm = pretrained_model(args.model)
    vlm.model.eval()

    if args.model == "qwen":
        set_collate_fn = qwen_collate_fn

    elif args.model == "llama":
        set_collate_fn = llama_collate_fn

    elif args.model == "gemma":
        set_collate_fn = gemma_collate_fn

    low_loader = DataLoader(
        test_low_dataset,
        batch_size = args.batch,
        shuffle = False,
        num_workers = 4,
        collate_fn = partial(set_collate_fn, processor = vlm.processor)
    )

    high_loader = DataLoader(
        test_high_dataset,
        batch_size = args.batch,
        shuffle = False,
        num_workers = 4,
        collate_fn = partial(set_collate_fn, processor = vlm.processor)
    )

    for batch_inputs, questions, answers in tqdm(low_loader, desc = f"{args.model} Inference (Low Resolution)"):
        batch_inputs = {k: v.to(vlm.model.device) if hasattr(v, "to") else v
                        for k, v in batch_inputs.items()}
        with torch.no_grad():
            gen_ids = vlm.model.generate(**batch_inputs, max_new_tokens=64)

        vlm.answer_dict[args.model]["low"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens=True))

        break

    for batch_inputs, questions, answers in tqdm(high_loader, desc = f"{args.model} Inference (High Resolution)"):
        batch_inputs = {k: v.to(vlm.model.device) if hasattr(v, "to") else v
                        for k, v in batch_inputs.items()}
        with torch.no_grad():
            gen_ids = vlm.model.generate(**batch_inputs, max_new_tokens=64)

        vlm.answer_dict[args.model]["high"].append(vlm.processor.batch_decode(gen_ids, skip_special_tokens=True))

        break

    # 결과 저장
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/{args.model}_results.json", "w") as f:
        json.dump(vlm.answer_dict[args.model], f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, required = True, default = "qwen")
    parser.add_argument("--batch", type = int, required = True, default = 32)
    parser.add_argument("--seed", type = int, required = True, default = 42)
    args = parser.parse_args()

    main(args)
