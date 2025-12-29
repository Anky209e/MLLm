# TODO: add finetune functions

import torch
from download_gpt import download_and_load_gpt2
from gpt import GPTModel
import numpy as np

import gpt

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Shortened context length (orig: 1024)
    "embedding_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.15,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}


settings, params = download_and_load_gpt2("124M", "downloaded_models")
print(settings)
print(params.keys())

gpt_model = GPTModel(cfg=GPT_CONFIG_124M)
gpt_model.eval()


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape Mismatch. Left{left.shape}, Right:{right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt_model=gpt_model, params=params):
    # Loading Embeddings
    gpt_model.position_embeddings.weight = assign(
        gpt_model.position_embeddings.weight, params["wpe"]
    )
    gpt_model.token_embeddings.weight = assign(
        gpt_model.token_embeddings.weight, params["wte"]
    )

    for b in range(len(params["blocks"])):
        # Loading Weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt_model.transformer_blocks[b].multi_head_attention.W_query.weight = assign(
            gpt_model.transformer_blocks[b].multi_head_attention.W_query.weight, q_w.T
        )
        gpt_model.transformer_blocks[b].multi_head_attention.W_key.weight = assign(
            gpt_model.transformer_blocks[b].multi_head_attention.W_key, k_w.T
        )
        gpt_model.transformer_blocks[b].multi_head_attention.W_value.weight = assign(
            gpt_model.transformer_blocks[b].multi_head_attention.W_value.weight, v_w.T
        )
        # Loading Bias
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt_model.transformer_blocks[b].multi_head_attention.W_query.bias = assign(
            gpt_model.transformer_blocks[b].multi_head_attention.W_query.bias, q_b
        )
        gpt_model.transformer_blocks[b].multi_head_attention.W_key.bias = assign(
            gpt_model.transformer_blocks[b].multi_head_attention.W_key.bias, k_b
        )
        gpt_model.transformer_blocks[b].multi_head_attention.W_value.bias = assign(
            gpt_model.transformer_blocks[b].multi_head_attention.W_value.bias, v_b
        )
        # Loading Output Layer Weights and Biases
        gpt_model.transformer_blocks[
            b
        ].multi_head_attention.output_projection.weight = assign(
            gpt_model.transformer_blocks[
                b
            ].multi_head_attention.output_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt_model.transformer_blocks[
            b
        ].multi_head_attention.output_projection.bias = assign(
            gpt_model.transformer_blocks[b].multi_head_attention.output_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"].T,
        )
        # Loading Weights and bias for Feed Forward network
        gpt_model.transformer_blocks[b].feed_forward_network.layers[0].weight = assign(
            gpt_model.transformer_blocks[b].feed_forward_network.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt_model.transformer_blocks[b].feed_forward_network.layers[0].bias = assign(
            gpt_model.transformer_blocks[b].feed_forward_network.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"].T,
        )
        gpt_model.transformer_blocks[b].feed_forward_network.layers[2].weight = assign(
            gpt_model.transformer_blocks[b].feed_forward_network.layers[2].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt_model.transformer_blocks[b].feed_forward_network.layers[2].bias = assign(
            gpt_model.transformer_blocks[b].feed_forward_network.layers[2].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"].T,
        )
        # Loading scale and shift for normalisation layers
        gpt_model.transformer_blocks[b].layer_norm1.scale = assign(
            gpt_model.transformer_blocks[b].layer_norm1.scale,
            params["blocks"][b]["ln_1"]["g"],
        )
        gpt_model.transformer_blocks[b].layer_norm1.shift = assign(
            gpt_model.transformer_blocks[b].layer_norm1.shift,
            params["blocks"][b]["ln_1"]["b"],
        )
        gpt_model.transformer_blocks[b].layer_norm2.scale = assign(
            gpt_model.transformer_blocks[b].layer_norm2.scale,
            params["blocks"][b]["ln_1"]["g"],
        )
        gpt_model.transformer_blocks[b].layer_norm2.shift = assign(
            gpt_model.transformer_blocks[b].layer_norm2.shift,
            params["blocks"][b]["ln_1"]["b"],
        )

        # Loading for final norm layer and logits
        gpt_model.layer_norm.scale = assign(gpt_model.layer_norm.scale, params["g"])
        gpt_model.layer_norm.shift = assign(gpt_model.layer_norm.shift, params["b"])

        gpt_model.output_layer.weight = assign(
            gpt_model.output_layer.weight, params["wte"]
        )
    return True
