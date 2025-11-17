from multi_head_attention import MultiHeadAttention
from layer_normalisation import LayerNorm
import torch


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
    norm_layer = LayerNorm(embedding_dim=dim_out)
    mheadatten = MultiHeadAttention(
        dim_in, dim_out, context_length, dropout=0.2, num_heads=2, qkv_bias=False
    )

    output = mheadatten(batch)

    print(output.shape)
    print(output)

    normed_output = norm_layer(output)
    print(normed_output)
    print(normed_output.var(dim=-1))
