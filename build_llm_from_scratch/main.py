import tiktoken
import torch

from gpt import GPTModel
from utils import generate_text_simple, get_model_details

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "embedding_dim": 768,
        "n_heads": 12,  # no of attention heads
        "n_layers": 12,  # no of transformer blocks
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    model = GPTModel(cfg=GPT_CONFIG_124M)
    params, total_size = get_model_details(model)
    print(f"Total number of Parameters:{params:,}")
    print(f"Total Size of Model:{total_size:.2f}MB")
    # out = model(batch)
    model.eval()
    idx = generate_text_simple(model, batch, 6, GPT_CONFIG_124M["context_length"])
    print(idx)
    # print(out.shape)
    # print(out)
