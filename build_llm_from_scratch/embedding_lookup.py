import torch

vocab_size = 5
embedding_dim = 5

torch.manual_seed(29)

embedding_layer = torch.nn.Embedding(
    num_embeddings=vocab_size, embedding_dim=embedding_dim
)

print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3, 1, 4])))
