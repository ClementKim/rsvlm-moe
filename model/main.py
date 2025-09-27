import torch
import argparse

from model.CLIP_Encoder import CLIPEncoder
from model.Transformer_Block import TransformerBlock


def build_models(embed_dim: int, mlp_dim: int, num_heads: int, num_experts: int, top_k: int | None, device: str):
    """CLIP 기반 멀티모달 임베딩 + TransformerBlock(MoE) 구성."""
    clip_model = CLIPEncoder(embed_dim=embed_dim).to(device)
    transformer = TransformerBlock(
        hidden_size=embed_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k
    ).to(device)
    return clip_model, transformer


def forward_example(clip_model, transformer, batch_size: int, seq_len: int, device: str):
    """더미 입력으로 end-to-end forward 수행.

    반환:
        out: (B, 2, H) Transformer 블록 출력
        pooled: (B, H) 두 토큰 평균 풀링
    주의: 실제 사용 시 적절한 전처리/토크나이저/이미지 resize 필요.
    """
    # 더미 이미지 (ViT-B/32 기본 입력 크기 224)
    images = torch.randn(batch_size, 3, 224, 224, device=device)

    # 더미 텍스트 토큰 (roberta-base vocab 크기 50265)
    input_ids = torch.randint(0, 50265, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    clip_model.eval()
    transformer.eval()
    with torch.no_grad():
        # 개별 모달 임베딩 (B, H)
        image_feat = clip_model.vision_projection(clip_model.vision_encoder(images))
        text_feat = clip_model.text_projection(clip_model.text_encoder(input_ids, attention_mask))

        # 두 토큰을 시퀀스로 결합 (B, 2, H)
        seq = torch.stack([image_feat, text_feat], dim=1)

        out = transformer(seq)  # (B, 2, H)
        pooled = out.mean(dim=1)

    return out, pooled


def main():
    parser = argparse.ArgumentParser(description="Run CLIP + TransformerBlock (MoE) forward example")
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--mlp-dim', type=int, default=2048)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=None, help='Top-k gating (None이면 dense)')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=8, help='텍스트 토큰 길이 (CLS 포함 가정)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    clip_model, transformer = build_models(
        embed_dim=args.embed_dim,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        device=device
    )

    out, pooled = forward_example(
        clip_model=clip_model,
        transformer=transformer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device
    )

    print(f"[SHAPE] Transformer output: {out.shape} (B, 2, H)")
    print(f"[SHAPE] Pooled representation: {pooled.shape} (B, H)")
    print("[DONE] Forward example completed.")


if __name__ == "__main__":
    main()