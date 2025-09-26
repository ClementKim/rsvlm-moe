import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 mlp_dim: int, 
                 out_dim: int | None = None, 
                 dropout_rate: float = 0.1, 
                 bias_std: float = 1e-6, 
                 use_bias: bool = True, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.out_dim = hidden_size if out_dim is None else out_dim
        self.dropout_rate = dropout_rate
        self._target_dtype = dtype

        self.fc1 = nn.Linear(hidden_size, mlp_dim, bias=use_bias)
        self.fc2 = nn.Linear(mlp_dim, self.out_dim, bias=use_bias)

        self.dropout = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        if use_bias:
            nn.init.normal_(self.fc1.bias, std=bias_std)
            nn.init.normal_(self.fc2.bias, std=bias_std)

    def forward(self, x, deterministic: bool | None = None):
        original_shape = x.shape

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self._target_dtype is not None:
            x = x.to(self._target_dtype)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self._maybe_dropout(x, deterministic)

        x = self.fc2(x)
        x = self._maybe_dropout(x, deterministic)

        if len(original_shape) == 2:
            x = x.squeeze(1)

        return x
    
    def _maybe_dropout(self, tensor, deterministic: bool | None):
        if deterministic is True:
            return tensor
        
        if deterministic is None:
            return self.dropout(tensor)
        
        return self.dropout(tensor)