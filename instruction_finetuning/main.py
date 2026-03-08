from functools import partial

import tiktoken
import torch
from dataset import InstructionDataset
from download_gpt import download_and_load_gpt2
from download_instruction_dataset import download_and_load_file
from finetune import load_weights_into_gpt
from gpt import GPTModel
from torch.utils.data import DataLoader
from utils import collate

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = (
        len(data) - train_portion - test_portion
    )  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---Using {device}---")
    cutomized_collate = partial(collate, device=device, allowed_max_length=1024)

    NUM_WORKERS = 0
    BATCH_SIZE = 6

    train_dataset = InstructionDataset(data=train_data, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=cutomized_collate,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    val_dataset = InstructionDataset(data=val_data, tokenizer=tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=cutomized_collate,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    test_dataset = InstructionDataset(data=test_data, tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=cutomized_collate,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
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

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    settings, params = download_and_load_gpt2("124M", "downloaded_models")
    model = GPTModel(cfg=BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    print("---Loaded Weights---")
