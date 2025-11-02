from qwen_vl_utils import process_vision_info

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