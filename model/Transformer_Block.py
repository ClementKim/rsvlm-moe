import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.1, attention_dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim = hidden_size,
            num_heads = num_heads,
            dropout = attention_dropout_rate,
            batch_first = True
        )

        self.attn_dropout = nn.Dropout(dropout_rate)

        # MoE Block
        # fill this part after implementing MoE block

    def forward(self, x, attn_mask = None, key_padding_mask = None):
        # Attention block
        residual = x

        x_norm = self.ln1(x)
        attn_out, attn_weights = self.attention(
            x_norm, x_norm, x_norm,
            attn_mask = attn_mask,
            key_padding_mask = key_padding_mask,
            need_weights = False
        )

        attn_out = self.attn_dropout(attn_out)
        x = residual + attn_out

        x_norm2 = self.ln2(x)

        # MoEout = self.moe_block(x_norm2)

        # x = x + MoEout

        # return x

        