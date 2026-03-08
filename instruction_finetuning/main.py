import torch
from utils import collate
from functools import partial
from dataset import InstructionDataset
from torch.utils.data import DataLoader
import tiktoken
from download_instruction_dataset import download_and_load_file


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
