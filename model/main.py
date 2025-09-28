import torch
import argparse

from CLIP_Encoder import CLIPEncoder
from Transformer_Block import TransformerBlock

def build_models(embed_dim: int, mlp_dim: int, num_heads: int, num_experts: int, top_k: int | None, device: str):
    """
    CLIP 인코더와 MoE 트랜스포머 블록을 생성하고 지정된 장치로 이동시킵니다.
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
    return clip_model, transformer

def forward_example(clip_model, transformer, batch_size: int, seq_len: int, device: str):
    """
    더미(dummy) 입력을 사용하여 모델의 전체 순전파 과정을 테스트합니다.
    실제 사용 시에는 적절한 이미지 전처리 및 텍스트 토크나이저가 필요합니다.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - out (torch.Tensor): 트랜스포머 블록의 최종 출력 (B, 2, H).
            - pooled (torch.Tensor): 이미지와 텍스트 토큰의 평균 풀링 결과 (B, H).
    """
    # 더미 이미지 생성 (ViT-B/32의 기본 입력 크기: 224x224)
    images = torch.randn(batch_size, 3, 224, 224, device=device)

    # 더미 텍스트 토큰 생성 (roberta-base의 vocab 크기: 50265)
    input_ids = torch.randint(0, 50265, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    # 모델을 평가 모드로 설정
    clip_model.eval()
    transformer.eval()
    with torch.no_grad():
        # 1. CLIP 인코더를 통해 이미지와 텍스트를 임베딩하고 하나의 시퀀스로 결합
        seq = clip_model(images, input_ids, attention_mask)  # (B, 2, H)

        # 2. 트랜스포머 블록을 통해 시퀀스 처리
        out = transformer(seq)  # (B, 2, H)
        
        # 3. 간단한 풀링 예시: 시퀀스 차원(dim=1)에 대해 평균 풀링
        pooled = out.mean(dim=1)

    return out, pooled

def main():
    """
    메인 실행 함수. 커맨드 라인 인자를 파싱하고, 모델을 빌드하며, 순전파 예제를 실행합니다.
    """
    parser = argparse.ArgumentParser(description="CLIP + TransformerBlock (MoE) 순전파 예제 실행")
    parser.add_argument('--embed-dim', type=int, default=512, help='임베딩 차원')
    parser.add_argument('--mlp-dim', type=int, default=2048, help='MLP의 중간 차원')
    parser.add_argument('--num-heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--num-experts', type=int, default=8, help='MoE 전문가 수')
    parser.add_argument('--top-k', type=int, default=None, help='MoE Top-k 게이팅 (None이면 dense)')
    parser.add_argument('--batch-size', type=int, default=2, help='배치 크기')
    parser.add_argument('--seq-len', type=int, default=8, help='텍스트 토큰 시퀀스 길이')
    args = parser.parse_args()

    # 사용 가능한 장치 설정 (CUDA 우선)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] 사용 장치: {device}")

    # 모델 빌드
    clip_model, transformer = build_models(
        embed_dim=args.embed_dim,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        device=device
    )

    # 모델 파라미터 수 계산 및 출력
    clip_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    transformer_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"[PARAMS] CLIP Encoder: {clip_params/1e6:.2f}M")
    print(f"[PARAMS] Transformer Block: {transformer_params/1e6:.2f}M")
    print(f"[PARAMS] Total: {(clip_params + transformer_params)/1e6:.2f}M")

    # 순전파 예제 실행
    out, pooled = forward_example(
        clip_model=clip_model,
        transformer=transformer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device
    )

    # 결과 텐서의 형태 출력
    print(f"[SHAPE] Transformer 출력: {out.shape} (B, 2, H)")
    print(f"[SHAPE] 풀링된 표현: {pooled.shape} (B, H)")
    print("[DONE] 순전파 예제 완료.")

if __name__ == "__main__":
    main()
