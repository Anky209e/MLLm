import torch.nn as nn
from gelu import GELU


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, embedding_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        output = self.layers(x)

        return output
