import tiktoken
import torch
import os

from gpt import GPTModel
from utils import (
    generate_text_simple,
    get_model_details,
    calc_loss_loader,
    calc_loss_batch,
)
from verdict_dataloader import VerdictDatasetV1, create_dataloader_v1

FILE_PATH = "data.txt"
TRAIN_RATION = 0.90
BATCH_SIZE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{DEVICE}")

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "embedding_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}

with open(FILE_PATH, "r", encoding="utf-8") as f:
    txt = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

total_characters = len(txt)
total_tokens = len(tokenizer.encode(txt))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

split_idx = int(TRAIN_RATION * len(txt))
train_data = txt[:split_idx]
val_data = txt[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=BATCH_SIZE,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=BATCH_SIZE,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

# Sanity check
if total_tokens * (TRAIN_RATION) < GPT_CONFIG_124M["context_length"]:
    print(
        "Not enough tokens for the training loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "increase the `training_ratio`"
    )

if total_tokens * (1 - TRAIN_RATION) < GPT_CONFIG_124M["context_length"]:
    print(
        "Not enough tokens for the validation loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "decrease the `training_ratio`"
    )

model = GPTModel(GPT_CONFIG_124M)
model.to(device=DEVICE)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, DEVICE)
    val_loss = calc_loss_loader(val_loader, model, DEVICE)

print(f"train_loss:{train_loss}")
print(f"val_loss:{val_loss}")
