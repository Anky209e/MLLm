import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 1e-5  # incase variance is zero
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        normed_x = (x - mean) / torch.sqrt(variance + self.eps)
        weighted_normed_x = self.scale * normed_x + self.shift
        return weighted_normed_x
