import torch
import torch.nn as nn

from layer_normalisation import LayerNorm
from transformer import Transformer


class GPTModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.token_embeddings = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embeddings = nn.Embedding(
            cfg["context_length"], cfg["embedding_dim"]
        )
        self.dropout_embeddings = nn.Dropout(cfg["drop_rate"])

        # transformer blocks
        self.tranformer_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_layers"])]
        )

        # normalisation layer
        self.layer_norm = LayerNorm(cfg["embedding_dim"])

        # Final linear layer which gives ouput of whole vocab vocab_size
        self.output_layer = nn.Linear(
            in_features=cfg["embedding_dim"], out_features=cfg["vocab_size"], bias=False
        )

    def forward(self, input_idx):
        batch_size, seq_length = input_idx.shape
        token_embeddings = self.token_embeddings(input_idx)
        position_embeddings = self.position_embeddings(
            torch.arange(seq_length, device=input_idx.device)
        )

        # adding position_embeddings to token_embeddings
        x = position_embeddings + token_embeddings
        x = self.dropout_embeddings(x)
        x = self.tranformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.output_layer(x)
        return logits
