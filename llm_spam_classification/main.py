import torch
from dataset import SpamDataset
import pandas as pd
import tiktoken
from torch.utils.data import DataLoader
from download_gpt import download_and_load_gpt2
from finetune import load_weights_into_gpt
from gpt import GPTModel

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
    csv_file="sms_data/train.csv", max_length=None, tokenizer=tokenizer
)
val_dataset = SpamDataset(
    csv_file="sms_data/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer,
)
test_dataset = SpamDataset(
    csv_file="sms_data/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer,
)

num_workers = 0
batch_size = 8

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Shortened context length (orig: 1024)
    "embedding_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.15,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings, params = download_and_load_gpt2("124M", "downloaded_models")
print(settings)
print(params.keys())

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)


settings, params = download_and_load_gpt2("124M", "downloaded_models")

model = GPTModel(cfg=BASE_CONFIG)
load_weights_into_gpt(model, params)
print("weights Loaded")
model.eval()
print(model)
