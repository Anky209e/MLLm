import tiktoken
import torch
from torch.utils.tensorboard import SummaryWriter

from gpt import GPTModel
from utils import (
    get_model_details,
    plot_losses,
    train_model_simple,
)
from verdict_dataloader import create_dataloader_v1

FILE_PATH = "data.txt"
FILE2 = "moby_dik.txt"

TRAIN_RATIO = 0.90
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{DEVICE}")
start_context = "So close behind some promontory "
writer = SummaryWriter("runs/gpt163m")
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "embedding_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.15,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}

with open(FILE_PATH, "r", encoding="utf-8") as f:
    txt1 = f.read()
with open(FILE2, "r", encoding="utf-8") as f:
    txt2 = f.read()
txt = txt1 + " <|endoftext|> " + txt2
tokenizer = tiktoken.get_encoding("gpt2")

total_characters = len(txt)
total_tokens = len(tokenizer.encode(txt, allowed_special={"<|endoftext|>"}))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

split_idx = int(TRAIN_RATIO * len(txt))
train_data = txt[:split_idx]
val_data = txt[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=BATCH_SIZE,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=BATCH_SIZE,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=True,
    num_workers=0,
)

# Sanity check
if total_tokens * (TRAIN_RATIO) < GPT_CONFIG_124M["context_length"]:
    print(
        "Not enough tokens for the training loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "increase the `training_ratio`"
    )

if total_tokens * (1 - TRAIN_RATIO) < GPT_CONFIG_124M["context_length"]:
    print(
        "Not enough tokens for the validation loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "decrease the `training_ratio`"
    )

model = GPTModel(GPT_CONFIG_124M)
model.to(device=DEVICE)
params, size = get_model_details(model)
print(f"Total Params:{params:,}")
print(f"Total Size:{size:.2f}MB")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    eval_freq=10,
    eval_iter=10,
    start_context=start_context,
    tokenizer=tokenizer,
    device=DEVICE,
    cfg=GPT_CONFIG_124M,
    writer=writer,
)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
