import torch
from transformers import pipeline

def llama_1b():
    # Llama 3.2 1B Instruct
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline("text-generation",
                    model = model_id,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto")
    
    messages = [
        {"role": "system", "content": "You are a priate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"}
    ]

def llama_7B():
    # Llama 3.2 7B Instruct
    pass