import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout=0.2, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Buffer for model state
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, input_dim = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2)

        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        attention_weights = self.dropout(attention_weights)
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
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)

    context_length = batch.shape[1]
    dim_in = inputs.shape[1]
    dim_out = 3

    attentionCausal = CausalAttention(
        dim_in=dim_in, dim_out=dim_out, context_length=context_length, dropout=0.0
    )
    context_vectors = attentionCausal(batch)
    print(context_vectors)
