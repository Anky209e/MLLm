import torch
from causal_attention import CausalAttention
import torch.nn as nn


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self, dim_in, dim_out, context_length, droput, num_heads, qkv_bias=False
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(dim_in, dim_out, context_length, droput, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


if __name__ == "__main__":
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # Journey
            [0.57, 0.85, 0.64],  # Starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],  # step
        ]
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)

    context_length = batch.shape[1]
    dim_in = inputs.shape[1]
    dim_out = 3

    mheadatten = MultiHeadAttentionWrapper(
        dim_in, dim_out, context_length, droput=0.2, num_heads=2, qkv_bias=False
    )

    output = mheadatten(batch)

    print(output.shape)
    print(output)
