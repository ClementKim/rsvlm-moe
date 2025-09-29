import torch
import argparse
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from CLIP_Encoder import CLIPEncoder
from Transformer_Block import TransformerBlock

def build_models(embed_dim: int, mlp_dim: int, num_heads: int, num_experts: int, top_k: int | None, device: str):
    """
    CLIP 인코더, MoE 트랜스포머 블록, Llama 3.2 디코더 및 프로젝션 레이어를 생성하고 지정된 장치로 이동시킵니다.
    """
    # CLIP 기반 멀티모달 인코더 모델 생성
    clip_model = CLIPEncoder(embed_dim=embed_dim).to(device)
    
    # MoE를 사용하는 트랜스포머 블록 모델 생성
    transformer = TransformerBlock(
        hidden_size=embed_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k
    ).to(device)

    # Llama 3.2 1B 모델 및 토크나이저 로드
    llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # Llama의 hidden size 가져오기
    llama_hidden_size = llama_model.config.hidden_size

    # Transformer 출력 차원을 Llama의 임베딩 차원으로 매핑하는 프로젝션 레이어
    projection = nn.Linear(embed_dim, llama_hidden_size).to(device)

    return clip_model, transformer, llama_model, llama_tokenizer, projection

def forward_and_generate(clip_model, transformer, llama_model, llama_tokenizer, projection, batch_size: int, seq_len: int, device: str):
    """
    더미 입력을 사용하여 전체 모델의 순전파 및 텍스트 생성 과정을 테스트합니다.
    """
    # 더미 이미지 생성
    images = torch.randn(batch_size, 3, 224, 224, device=device)

    # 더미 텍스트 토큰 생성
    input_ids = torch.randint(0, 50265, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    # 모델을 평가 모드로 설정
    clip_model.eval()
    transformer.eval()
    llama_model.eval()
    projection.eval()

    with torch.no_grad():
        # 1. CLIP 인코더와 Transformer 블록을 통해 시퀀스 처리
        seq = clip_model(images, input_ids, attention_mask)
        transformer_out = transformer(seq)  # (B, 2, embed_dim)
        
        # 2. Transformer 출력을 Llama의 임베딩 공간으로 프로젝션
        projected_embeds = projection(transformer_out)  # (B, 2, llama_hidden_size)

        # 3. Llama 모델을 사용하여 텍스트 생성
        # 생성된 텍스트의 시작을 알리는 프롬프트 추가 (선택 사항)
        # 여기서는 간단히 프로젝션된 임베딩만 사용
        generated_ids = llama_model.generate(
            inputs_embeds=projected_embeds.to(llama_model.dtype),
            max_new_tokens=50,
            eos_token_id=llama_tokenizer.eos_token_id
        )
        
        # 생성된 ID를 텍스트로 디코딩
        generated_text = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text

def main():
    """
    메인 실행 함수. 커맨드 라인 인자를 파싱하고, 모델을 빌드하며, 순전파 및 생성 예제를 실행합니다.
    """
    parser = argparse.ArgumentParser(description="CLIP + Transformer (MoE) + Llama 3.2 순전파 및 생성 예제")
    parser.add_argument('--embed-dim', type=int, default=512, help='CLIP 및 Transformer의 임베딩 차원')
    parser.add_argument('--mlp-dim', type=int, default=2048, help='MLP의 중간 차원')
    parser.add_argument('--num-heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--num-experts', type=int, default=8, help='MoE 전문가 수')
    parser.add_argument('--top-k', type=int, default=None, help='MoE Top-k 게이팅 (None이면 dense)')
    parser.add_argument('--batch-size', type=int, default=1, help='배치 크기 (생성에는 1로 설정 권장)')
    parser.add_argument('--seq-len', type=int, default=8, help='텍스트 토큰 시퀀스 길이')
    args = parser.parse_args()

    # 사용 가능한 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] 사용 장치: {device}")

    # 모델 빌드
    clip_model, transformer, llama_model, llama_tokenizer, projection = build_models(
        embed_dim=args.embed_dim,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        device=device
    )

    # 모델 파라미터 수 계산
    clip_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    transformer_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    llama_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    projection_params = sum(p.numel() for p in projection.parameters() if p.requires_grad)
    total_params = clip_params + transformer_params + llama_params + projection_params
    
    print(f"[PARAMS] CLIP Encoder: {clip_params/1e6:.2f}M")
    print(f"[PARAMS] Transformer Block: {transformer_params/1e6:.2f}M")
    print(f"[PARAMS] Llama 3.2 1B: {llama_params/1e6:.2f}M")
    print(f"[PARAMS] Projection Layer: {projection_params/1e6:.2f}M")
    print(f"[PARAMS] Total: {total_params/1e6:.2f}M")

    # 순전파 및 생성 예제 실행
    generated_text = forward_and_generate(
        clip_model=clip_model,
        transformer=transformer,
        llama_model=llama_model,
        llama_tokenizer=llama_tokenizer,
        projection=projection,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device
    )

    # 결과 출력
    print("\n[GENERATED TEXT]")
    for i, text in enumerate(generated_text):
        print(f"--- Sample {i+1} ---")
        print(text)
    
    print("\n[DONE] 순전파 및 생성 예제 완료.")

if __name__ == "__main__":
    main()
