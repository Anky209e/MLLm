import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, dim_in, dim_out, context_length, dropout, num_heads, qkv_bias=False
    ) -> None:
        super().__init__()

        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads

        self.W_query = nn.Linear(dim_in, dim_out, qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Splitting the weight matrix by head_dim
        # (batch_size,num_tokens,dim_out) --> (batch_size,num_tokens,num_heads,head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose (b,num_token,num_heads,head_dim) --> (b,num_heads,num_tokens,head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # attention scores
        attention_scores = queries @ keys.transpose(2, 3)  # dot product

        mask = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask, -torch.inf)
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vectors = (attention_weights @ values).transpose(
            1, 2
        )  # (b,num_tokens,num_heads,head_dim)

        # now we need to combine heads of our all vectors generated for each head_dim
        # self.d_out = self.num_heads * self.head_dim

        context_vectors = context_vectors.contiguous().view(
            batch_size, num_tokens, self.dim_out
        )

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
    dim_out = 6
    num_heads = 2

    mheadatten = MultiHeadAttention(
        dim_in, dim_out, context_length, dropout=0.2, num_heads=2, qkv_bias=False
    )

    output = mheadatten(batch)

    print(output.shape)
    print(output)
