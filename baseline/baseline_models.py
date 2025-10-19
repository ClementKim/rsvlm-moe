import torch

# models
from transformers import Qwen2_5_VLForConditionalGeneration # Qwen2.5-VL-Instruct 모델용
from transformers import MllamaForConditionalGeneration     # Llama-3.2-Vision-Instruct 모델용
from transformers import Gemma3ForConditionalGeneration     # gemma-3-it 모델용
from transformers import AutoProcessor, AutoModelForImageTextToText # BLIP2
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration # InstructBLIP

# processors
from transformers import AutoProcessor # 자동 프로세서 임포트

# qwen vision language utils
from qwen_vl_utils import process_vision_info # qwen 비전-언어 유틸리티 임포트

## fucntions for models and processors
def qwen_vl(param: int = 72):  # qwen_vl 함수 정의
    
    if param == 72:
        model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    
    elif param == 32:
        model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

    elif param == 7:
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    elif param == 3:
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # 1) Processor 먼저 로드
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
    )

    # 2) Decoder-only에서 필수: 왼쪽 패딩 강제 + pad_token 설정
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # Qwen 계열은 eos를 pad로 재사용

    # 3) 모델 로드
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # 4) 모델/생성 설정에 pad_token_id 동기화
    pad_id = tok.pad_token_id
    if getattr(model.config, "pad_token_id", None) != pad_id:
        model.config.pad_token_id = pad_id
    if hasattr(model, "generation_config") and model.generation_config.pad_token_id != pad_id:
        model.generation_config.pad_token_id = pad_id

    model.eval()

    return model, processor  # 모델과 프로세서 반환

def llama_vision(param : int = 90): # llama_vision 함수 정의
    if param == 90:
        model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct" # 모델 이름 설정

    elif param == 11:
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(model_name, dtype = torch.bfloat16, device_map = "auto").eval() # 사전 학습된 모델 로드
    processor = AutoProcessor.from_pretrained(model_name) # 사전 학습된 프로세서 로드

    return model, processor # 모델과 프로세서 반환

def gemma(param : int = 27): # gemma 함수 정의
    '''
    gemma-3-27b-it model
    '''

    if param == 27:
        model_name = "google/gemma-3-27b-it" # 모델 이름 설정

    elif param == 12:
        model_name = "google/gemma-3-12b-it"

    elif param == 4:
        model_name = "google/gemma-3-4b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, device_map = "auto").eval() # 사전 학습된 모델 로드 후 평가 모드로 설정
    processor = AutoProcessor.from_pretrained(model_name) # 사전 학습된 프로세서 로드

    return model, processor # 모델과 프로세서 반환

def blip2():
    '''
    blip2 model
    '''

    model_name = "Salesforce/blip2-opt-2.7b"

    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype = torch.float16, device_map = "auto").eval()
    processor = AutoProcessor.from_pretrained(model_name)

    tok = processor.tokenizer
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    if getattr(model.config, "pad_token_id", None) != tok.pad_token_id:
        model.config.pad_token_id = tok.pad_token_id

    return model, processor

def instructBLIP():
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")

    tok = processor.tokenizer
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    if getattr(model.config, "pad_token_id", None) != tok.pad_token_id:
        model.config.pad_token_id = tok.pad_token_id

    return model, processor