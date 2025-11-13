import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T
        # Scaling by root of embedding dimension of key to reduce variance
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vectors = attention_weights @ values
        return context_vectors


if __name__ == "__main__":
    torch.manual_seed(29)
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

    dim_in = inputs.shape[1]
    dim_out = 2

    s_attn = SelfAttention(dim_in=dim_in, dim_out=dim_out)

    print(s_attn(inputs))
