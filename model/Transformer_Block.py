import torch
import torch.nn as nn
import torch.nn.functional as F
from MoE_Block import MLPBlock

class TransformerBlock(nn.Module):
    """
    Mixture of Experts (MoE)를 피드포워드 네트워크로 사용하는 트랜스포머 블록.
    Pre-Layer Normalization 구조를 따릅니다.
    """
    def __init__(self,
                 hidden_size: int,
                 mlp_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 num_experts: int = 8,
                 top_k: int | None = None):
        """
        TransformerBlock을 초기화합니다.

        Args:
            hidden_size (int): 입력 및 은닉 상태의 차원.
            mlp_dim (int): 각 전문가(MLP)의 중간 계층 차원.
            num_heads (int): 멀티헤드 어텐션의 헤드 수.
            dropout_rate (float): 일반 드롭아웃 비율.
            attention_dropout_rate (float): 어텐션 가중치를 위한 드롭아웃 비율.
            num_experts (int): MoE 계층의 전문가(MLP) 수.
            top_k (int | None): 게이팅에서 사용할 상위 k개의 전문가. None이면 모든 전문가를 사용(dense).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # hidden_size는 num_heads로 나누어 떨어져야 합니다.
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        if self.top_k is not None:
            assert 1 <= self.top_k <= self.num_experts, "top_k must be within [1, num_experts]"

        # 어텐션 이전의 Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        # MoE 이전의 Layer Normalization
        self.ln2 = nn.LayerNorm(hidden_size)

        # 멀티헤드 셀프 어텐션
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True
        )
        # 어텐션 출력에 적용할 드롭아웃
        self.attn_dropout = nn.Dropout(dropout_rate)

        # MoE를 위한 전문가(MLP) 리스트
        self.experts = nn.ModuleList([
            MLPBlock(hidden_size, mlp_dim, out_dim=hidden_size, dropout_rate=dropout_rate)
            for _ in range(num_experts)
        ])

        # 게이팅 네트워크: 각 토큰이 어떤 전문가에게 할당될지 결정
        self.gate_proj = nn.Linear(hidden_size, num_experts)
        # MoE 출력에 적용할 드롭아웃
        self.moe_dropout = nn.Dropout(dropout_rate)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mixture of Experts (MoE) 순전파를 수행합니다.
        입력 토큰 각각에 대해 게이트가 전문가의 가중치를 계산하고, 가중합을 통해 출력을 생성합니다.
        """
        # 게이트 로짓 계산: (B, T, H) -> (B, T, E)
        logits = self.gate_proj(x)

        if self.top_k is None:
            # Dense Gating: 모든 전문가를 사용
            # 소프트맥스를 통해 전문가 가중치(확률) 계산
            probs = F.softmax(logits, dim=-1)  # (B, T, E)
            # 모든 전문가의 출력을 계산
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # (B, T, E, H)
            # 전문가 출력과 가중치를 곱하여 가중합을 계산
            moe_out = torch.sum(probs.unsqueeze(-1) * expert_outputs, dim=-2)  # (B, T, H)
        else:
            # Sparse Gating: Top-k 전문가만 사용
            # 가장 높은 로짓을 가진 top-k 전문가의 값과 인덱스를 찾음
            topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # (B, T, K)
            # top-k 로짓에 대해서만 소프트맥스를 적용하여 가중치 계산
            probs = F.softmax(topk_vals, dim=-1)  # (B, T, K)
            
            # 모든 전문가의 출력을 미리 계산 (최적화 가능 지점)
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # (B, T, E, H)
            
            # top-k 인덱스를 사용하여 해당 전문가의 출력만 선택
            # gather를 위해 인덱스 텐서의 차원을 확장
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)  # (B, T, K, H)
            selected_experts = torch.gather(expert_outputs, dim=-2, index=idx_expanded)  # (B, T, K, H)
            
            # 선택된 전문가들의 출력과 가중치를 곱하여 가중합 계산
            moe_out = torch.sum(probs.unsqueeze(-1) * selected_experts, dim=-2)  # (B, T, H)

        return self.moe_dropout(moe_out)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        트랜스포머 블록의 순전파를 수행합니다.
        (Pre-LN: LayerNorm -> Module -> Residual)
        """
        # 1. 멀티헤드 어텐션
        residual = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = residual + self.attn_dropout(attn_out)

        # 2. MoE 피드포워드 네트워크
        residual2 = x
        x_norm2 = self.ln2(x)
        moe_out = self._moe(x_norm2)
        x = residual2 + moe_out

        return x
