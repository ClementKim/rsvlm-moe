import torch # torch 라이브러리 임포트

# models
from transformers import Qwen2_5_VLForConditionalGeneration # Qwen2.5-VL-72B-Instruct 모델용
from transformers import MllamaForConditionalGeneration     # Llama-3.2-90B-Vision-Instruct 모델용
from transformers import Gemma3ForConditionalGeneration     # gemma-3-27b-it 모델용
from transformers import GenerationConfig                   # phi-4 멀티모달 모델용
from transformers import AutoModelForCausalLM               # Phi-4 멀티모달 모델용

# processors
from transformers import AutoProcessor # 자동 프로세서 임포트

# qwen vision language utils
from qwen_vl_utils import process_vision_info # qwen 비전-언어 유틸리티 임포트

## fucntions for models and processors
def qwen_vl(): # qwen_vl 함수 정의
    '''
    Qwen2.5-VL-72B-Instruct model
    '''
    
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct" # 모델 이름 설정
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto") # 사전 학습된 모델 로드
    processor = AutoProcessor.from_pretrained(model_name, min_pixels = 256*28*28, max_pixels = 1280*28*28) # 사전 학습된 프로세서 로드

    return model, processor # 모델과 프로세서 반환

def llama_vision(): # llama_vision 함수 정의
    '''
    Llama-3.2-90B-Vision-Instruct model
    '''

    model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct" # 모델 이름 설정

    model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype = torch.bfloat16, device_map = "auto") # 사전 학습된 모델 로드
    processor = AutoProcessor.from_pretrained(model_name) # 사전 학습된 프로세서 로드

    return model, processor # 모델과 프로세서 반환

def gemma(): # gemma 함수 정의
    '''
    gemma-3-27b-it model
    '''

    model_name = "google/gemma-3-27b-it" # 모델 이름 설정

    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, device_map = "auto").eval() # 사전 학습된 모델 로드 후 평가 모드로 설정
    processor = AutoProcessor.from_pretrained(model_name) # 사전 학습된 프로세서 로드

    return model, processor # 모델과 프로세서 반환

def phi4(): # phi4 함수 정의
    '''
    phi 4 multimodal model
    '''
    model_name = "microsoft/Phi-4-multimodal-instruct" # 모델 이름 설정

    model = AutoModelForCausalLM.from_pretrained(model_name, # 사전 학습된 모델 로드
                                                 revision = "main", # 메인 리비전 사용
                                                 device_map = "cuda", # CUDA 장치에 매핑
                                                 torch_dtype = "auto", # 토치 데이터 타입 자동 설정
                                                 trust_remote_code = True, # 원격 코드 신뢰
                                                 _attn_implementation = "flash_attention_2").cuda() # flash attention 2 사용 및 CUDA로 이동
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True) # 사전 학습된 프로세서 로드 (원격 코드 신뢰)

    generation_config = GenerationConfig.from_pretrained(model_name) # 사전 학습된 생성 설정 로드
    
    return model, processor, generation_config # 모델, 프로세서, 생성 설정 반환