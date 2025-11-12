import torch

# from torch.utils.data import dataloader
from verdict_dataloader import create_dataloader_v1

FILE = "Data.txt"

with open(FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
input, target = next(data_iter)

print(input.shape)

input_embedding = token_embedding_layer(input)

print(input_embedding.shape)

# Here our max length will be our context length which is 4 ie the number of total position
positional_embedding_layer = torch.nn.Embedding(max_length, embedding_dim=output_dim)

# Generating position embeddings for all 4 postions of token
positional_embeddings = positional_embedding_layer(torch.arange(max_length))

# final_input_embeddings[8x4x256] =  input_embedding[8x4x256] + positional_embeddings [4x256]
input_embedding = input_embedding + positional_embeddings

print(
    f"Final Input : {input_embedding.shape} = input_embedding : {input_embedding.size()} + positional_embeddings : {positional_embeddings.size()}"
)
