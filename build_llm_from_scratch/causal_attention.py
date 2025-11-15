import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_score = queries @ keys.T
        context_length = attention_score.shape[-1]
        # using -inf to avoide data leakage during softmax
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked_attention_scores = attention_score.masked_fill(mask.bool(), -torch.inf)
        attention_weights = torch.softmax(
            masked_attention_scores / masked_attention_scores.shape[-1] ** 0.5, dim=-1
        )

        context_vectors = attention_weights @ values
        return context_vectors


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
    dim_in = inputs.shape[1]
    dim_out = 2
    att_c = CausalAttention(dim_in, dim_out)
    print(att_c(inputs))
