import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

class MLPBlock(nn.Module):
    """
    표준 2계층 피드포워드 네트워크(FFN) 또는 다층 퍼셉트론(MLP) 블록.
    Transformer의 FFN과 동일한 구조(Linear -> Activation -> Dropout -> Linear -> Dropout)를 가집니다.
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
        """
        MLPBlock을 초기화합니다.

        Args:
            hidden_size (int): 입력 및 기본 은닉 차원.
            mlp_dim (int): 중간 계층의 확장 차원.
            out_dim (Optional[int]): 출력 차원. None이면 hidden_size와 동일하게 설정됩니다.
            dropout_rate (float): 드롭아웃 비율.
            bias_std (float): 편향(bias) 초기화를 위한 표준편차.
            use_bias (bool): 선형 계층에서 편향을 사용할지 여부.
            dtype (Optional[torch.dtype]): 내부 계산에 사용할 데이터 타입.
            activation (Optional[Callable]): 사용할 활성화 함수. None이면 F.gelu가 사용됩니다.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.out_dim = hidden_size if out_dim is None else out_dim
        self.dropout_rate = dropout_rate
        self._target_dtype = dtype
        self.activation = activation or F.gelu

        # 첫 번째 선형 계층 (입력 -> 중간 차원)
        self.fc1 = nn.Linear(hidden_size, mlp_dim, bias=use_bias)
        # 두 번째 선형 계층 (중간 차원 -> 출력 차원)
        self.fc2 = nn.Linear(mlp_dim, self.out_dim, bias=use_bias)
        # 드롭아웃 계층
        self.dropout = nn.Dropout(dropout_rate)

        # 가중치 및 편향 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if use_bias:
            nn.init.normal_(self.fc1.bias, std=bias_std)
            nn.init.normal_(self.fc2.bias, std=bias_std)

    def forward(self, x: torch.Tensor, deterministic: bool | None = None) -> torch.Tensor:
        """
        MLP 블록의 순전파를 수행합니다.

        Args:
            x (torch.Tensor): (B, ..., H) 형태의 입력 텐서.
            deterministic (bool | None): True이면 드롭아웃을 비활성화합니다.
                                         None이면 self.training 상태에 따라 드롭아웃이 적용됩니다.

        Returns:
            torch.Tensor: (B, ..., out_dim) 형태의 출력 텐서.
        """
        # 필요 시 데이터 타입을 변환합니다.
        if self._target_dtype is not None and x.dtype != self._target_dtype:
            x = x.to(self._target_dtype)

        # 첫 번째 선형 계층 -> 활성화 함수 -> 드롭아웃
        x = self.fc1(x)
        x = self.activation(x)
        x = self._maybe_dropout(x, deterministic)

        # 두 번째 선형 계층 -> 드롭아웃
        x = self.fc2(x)
        x = self._maybe_dropout(x, deterministic)

        return x

    def _maybe_dropout(self, tensor: torch.Tensor, deterministic: bool | None) -> torch.Tensor:
        """
        deterministic 플래그에 따라 조건부로 드롭아웃을 적용합니다.
        """
        # deterministic이 True이면 드롭아웃을 적용하지 않습니다.
        if deterministic:
            return tensor
        # self.training이 False이면 dropout 모듈은 자동으로 비활성화됩니다.
        return self.dropout(tensor)
