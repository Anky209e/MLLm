import tiktoken
import torch

from gpt import GPTModel

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
    for i, j in model.named_parameters():
        print(i, j.numel())
    params = sum(p.numel() for p in model.parameters())
    print(f"Total number of Parameters:{params:,}")
    total_bytes = params * 4
    total_size = total_bytes / (1024 * 1024)
    print(f"Total Size of Model:{total_size:.2f}MB")
    # out = model(batch)
    # print(out.shape)
    # print(out)
