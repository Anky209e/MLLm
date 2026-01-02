import torch
from dataset import SpamDataset
import pandas as pd
import tiktoken
from torch.utils.data import DataLoader

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

print("Train loader:")
for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
