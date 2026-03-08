import torch
from torch.utils.data import Dataset
from utils import format_to_alpaca


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Tokenizer
        self.encoded_texts = []
        for entry in data:
            alpaca_instruction = format_to_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = alpaca_instruction + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
