import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MoE_Block import MLPBlock  # 명확한 클래스명 사용

class TransformerBlock(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 mlp_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 num_experts: int = 8,
                 top_k: int | None = None,
                 moe_dropout_rate: float | None = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k  # 현재 구현은 dense softmax; top_k 설정 시 top-k만 사용

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        if self.top_k is not None:
            assert 1 <= self.top_k <= self.num_experts, "top_k must be within [1, num_experts]"

        # Layer Normalizations (Pre-LN 스타일)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout_rate)

        # Experts (MLP Blocks)
        self.experts = nn.ModuleList([
            MLPBlock(hidden_size, mlp_dim, out_dim=hidden_size, dropout_rate=dropout_rate)
            for _ in range(num_experts)
        ])

        # Gating projection
        self.gate_proj = nn.Linear(hidden_size, num_experts)

        # MoE 출력 드롭아웃 (선택)
        self.moe_dropout = nn.Dropout(moe_dropout_rate if moe_dropout_rate is not None else dropout_rate)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        # 게이트 로짓 & 확률
        logits = self.gate_proj(x)  # (B, T, E)

        # Top-k sparsity: 선택된 k 이외는 0 가중치
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # (B,T,K)
        probs = F.softmax(topk_vals, dim=-1)  # (B,T,K)
        # 전문가 출력 미리 전부 계산 (간단 구현) -> 필요 시 선택된 전문가만 계산하도록 최적화 가능
        expert_outputs = [expert(x) for expert in self.experts]  # 각 (B,T,H)
        stacked = torch.stack(expert_outputs, dim=-2)  # (B,T,E,H)
        # 가중합 구성
        # gather 인덱스 확장 준비
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)  # (B,T,K,H)
        selected = torch.gather(stacked, dim=-2, index=idx_expanded)  # (B,T,K,H)
        moe_out = torch.sum(probs.unsqueeze(-1) * selected, dim=-2)  # (B,T,H)

        return self.moe_dropout(moe_out)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Multi-Head Attention (Pre-LN)
        residual = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = residual + self.attn_dropout(attn_out)

        # MoE FeedForward
        residual2 = x
        x_norm2 = self.ln2(x)
        moe_out = self._moe(x_norm2)
        x = residual2 + moe_out

        return x