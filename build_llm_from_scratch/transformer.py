import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForwardNeuralNetwork
from gelu import GELU
from layer_normalisation import LayerNorm


class Transformer(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.multi_head_attention = MultiHeadAttention(
            dim_in=cfg["dim_in"],
            dim_out=cfg["dim_out"],
            context_length=cfg["context_length"],
            dropout=cfg["dropout"],
            num_heads=cfg["num_heads"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.feed_forward_network = FeedForwardNeuralNetwork(
            embedding_size=cfg["embedding_size"]
        )

        self.gelu = GELU()
        self.layer_norm1 = LayerNorm(embedding_dim=cfg["embedding_size"])
        self.layer_norm2 = LayerNorm(embedding_dim=cfg["embedding_size"])

        self.dropout = nn.Dropout(cfg["dropout"])

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
