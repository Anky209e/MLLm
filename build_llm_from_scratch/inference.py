import torch
import tiktoken
from gpt import GPTModel
from utils import generate_and_print_sample

WEIGHT_PATH = "weights_100ep.pth"
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "embedding_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_context = "It was not till three years later that, in the course of a few weeks'"

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device=device)
generate_and_print_sample(
    model=model,
    tokenizer=tokenizer,
    device=device,
    start_context=start_context,
    set_train=False,
)
