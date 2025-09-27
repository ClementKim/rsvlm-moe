import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

class MLPBlock(nn.Module):
    """두 단계 Feed-Forward (FFN) 블록.

    Transformer FFN과 동일한 구조 (Linear -> 활성함수 -> Dropout -> Linear -> Dropout).

    입력 형태:
        (B, H) 또는 (B, T, H). 2D 입력은 내부적으로 (B, 1, H)로 확장 후 다시 복구.

    Args:
        hidden_size: 입력/기본 히든 차원.
        mlp_dim: 확장 차원 (중간 피드포워드 차원).
        out_dim: 출력 차원 (기본: hidden_size).
        dropout_rate: 드롭아웃 비율.
        bias_std: bias 초기화 표준편차 (아주 작게 설정해 초반 영향 최소화).
        use_bias: Linear 계층 bias 사용 여부.
        dtype: 내부 연산 강제 dtype (ex. torch.float16).
        activation: 활성함수 (기본 GELU). 호출 가능한 함수 (tensor -> tensor).

    deterministic:
        forward 인자. True 이면 dropout 비활성. False/None 이면 일반 동작 (self.training에 따라 Dropout).
    """
    def __init__(self,
                 hidden_size: int,
                 mlp_dim: int,
                 out_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 bias_std: float = 1e-6,
                 use_bias: bool = True,
                 dtype: Optional[torch.dtype] = None,
                 activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.out_dim = hidden_size if out_dim is None else out_dim
        self.dropout_rate = dropout_rate
        self._target_dtype = dtype
        self.activation = activation or F.gelu

        self.fc1 = nn.Linear(hidden_size, mlp_dim, bias=use_bias)
        self.fc2 = nn.Linear(mlp_dim, self.out_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

        # 가중치 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if use_bias:
            nn.init.normal_(self.fc1.bias, std=bias_std)
            nn.init.normal_(self.fc2.bias, std=bias_std)

    def forward(self, x: torch.Tensor, deterministic: bool | None = None) -> torch.Tensor:
        original_shape = x.shape

        # (B, H) 입력을 (B, 1, H)로 확장하여 (B, T, H) 형태 처리 통일
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeezed = True

        # 필요 시 dtype 강제 변환
        if self._target_dtype is not None and x.dtype != self._target_dtype:
            x = x.to(self._target_dtype)

        # 첫 번째 선형 + 활성함수 + 드롭아웃
        x = self.fc1(x)
        x = self.activation(x)
        x = self._maybe_dropout(x, deterministic)

        # 두 번째 선형 + 드롭아웃
        x = self.fc2(x)
        x = self._maybe_dropout(x, deterministic)

        if squeezed:
            x = x.squeeze(1)
        return x

    def _maybe_dropout(self, tensor: torch.Tensor, deterministic: bool | None) -> torch.Tensor:
        # deterministic=True -> 항상 드롭아웃 비활성
        if deterministic:
            return tensor
        # Dropout 모듈은 self.training이 False면 자동 no-op 이므로 그대로 호출
        return self.dropout(tensor)