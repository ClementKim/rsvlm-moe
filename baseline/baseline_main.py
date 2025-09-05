import torch

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
    def __init__(self):
        self.answer_dict = {"qwen": {}, "llama": {}, "gemma": {}, "phi4": {}}

        self.qwen_model, self.qwen_processor = baseline_models.qwen_vl()
        # self.llama_model, self.llama_processor = baseline_models.llama_vision()
        # self.gemma_model, self.gemma_processor = baseline_models.gemma()

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
        messages = [
            {"role": "user", "content": ["<image>", q]}
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    vlm = pretrained_model()
    vlm.qwen_model.eval()
    # vlm.llama_model.eval()
    # vlm.gemma_model.eval()

    # 데이터 준비
    dataset_dict = preprocessing.rsvqa_all({})
    low_dataset = RS_dataset(dataset_dict, "low")
    high_dataset = RS_dataset(dataset_dict, "high")

    # 예시: Qwen용 DataLoader (필요한 모델만 골라서 사용)
    low_qwen_loader = DataLoader(
        low_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(qwen_collate_fn, processor=vlm.qwen_processor),
    )

    # low_llama_loader = DataLoader(
    #     low_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=partial(llama_collate_fn, processor=vlm.llama_processor),
    # )
    # low_gemma_loader = DataLoader(
    #     low_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=partial(gemma_collate_fn, processor=vlm.gemma_processor),
    # )

    for batch_inputs, questions, answers in tqdm(low_qwen_loader, desc = "Qwen Inference"):
        batch_inputs = {k: v.to(vlm.qwen_model.device) if hasattr(v, "to") else v
                        for k, v in batch_inputs.items()}
        with torch.no_grad():
            gen_ids = vlm.qwen_model.generate(**batch_inputs, max_new_tokens=64)

        print(vlm.qwen_processor.batch_decode(gen_ids, skip_special_tokens=True))
        break

    # for batch_inputs, questions, answers in tqdm(low_llama_loader, desc = "Llama Inference"):
    #     batch_inputs = {k: v.to(vlm.llama_model.device) if hasattr(v, "to") else v
    #                     for k, v in batch_inputs.items()}
        
    #     with torch.no_grad():
    #         gen_ids = vlm.llama_model.generate(**batch_inputs, max_new_tokens=64)

    #     print(vlm.llama_processor.batch_decode(gen_ids, skip_special_tokens=True))
    #     break

    # for batch_inputs, questions, answers in tqdm(low_gemma_loader, desc = "Gemma Inference"):
    #     batch_inputs = {k: v.to(vlm.gemma.device) if hasattr(v, "to") else v
    #                     for k, v in batch_inputs.items()}
        
    #     with torch.no_grad():
    #         gen_ids = vlm.gemma_model.generate(**batch_inputs, max_new_tokens=64)

    #     print(vlm.llama_processor.batch_decode(gen_ids, skip_special_tokens=True))
    #     break

if __name__ == "__main__":
    main()