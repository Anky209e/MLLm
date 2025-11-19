import torch.nn as nn

from feed_forward_network import FeedForwardNeuralNetwork
from gelu import GELU
from layer_normalisation import LayerNorm
from multi_head_attention import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.multi_head_attention = MultiHeadAttention(
            dim_in=cfg["embedding_dim"],
            dim_out=cfg["embedding_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.feed_forward_network = FeedForwardNeuralNetwork(
            embedding_dim=cfg["embedding_dim"]
        )

        self.gelu = GELU()
        self.layer_norm1 = LayerNorm(embedding_dim=cfg["embedding_dim"])
        self.layer_norm2 = LayerNorm(embedding_dim=cfg["embedding_dim"])

        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        residual = x
        out = self.layer_norm1(x)
        out = self.multi_head_attention(out)
        out = self.dropout(out)
        out = out + residual

        residual = out
        out = self.layer_norm2(out)
        out = self.feed_forward_network(out)
        out = self.dropout(out)
        out = out + residual

        return out
