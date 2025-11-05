import torch

from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
from MoAI.moai.load_moai import prepare_moai

image = Resize(size=(490, 490), antialias=False)(pil_to_tensor(Image.open(image_path)))

prompt = "Describe this image in detail."

moai_model, moai_processor, seg_model, seg_processor, od_model, od_processor, sgg_model, ocr_model = prepare_moai(moai_path='BK-Lee/MoAI-7B', bits=4, grad_ckpt=False, lora=False, dtype='fp16')

moai_inputs = moai_model.demo_process(image=image, 
                                    prompt=prompt, 
                                    processor=moai_processor,
                                    seg_model=seg_model,
                                    seg_processor=seg_processor,
                                    od_model=od_model,
                                    od_processor=od_processor,
                                    sgg_model=sgg_model,
                                    ocr_model=ocr_model,
                                    device='cuda:0')

with torch.inference_mode():
    generate_ids = moai_model.generate(**moai_inputs,
                                       do_sample=True,
                                       temperature=0.9,
                                       top_p=0.95,
                                       max_new_tokens=256,
                                       use_cache=True)
    
    answer = moai_processor.batch_decode(generate_ids, skip_special_tokens=True)[0].split('[U')[0]
    print(answer)