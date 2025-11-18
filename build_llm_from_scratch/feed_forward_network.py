import torch.nn as nn
from gelu import GELU


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, embedding_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.layer2 = nn.Linear(4 * embedding_size, embedding_size)
        self.gelu = GELU()

    def forward(self, x):
        output = self.layer1(x)
        output = self.gelu(output)
        output = self.layer2(output)

        return output
