import torch
import tiktoken
from gpt import GPTModel
from utils import text_to_token_ids, token_ids_to_text, generate


if __name__ == "__main__":
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

    start_context = (
        "It was not till three years later that, in the course of a few weeks'"
    )

    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.to(device=device)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer, device),
        max_new_tokens=50,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=50,
        temperature=1,
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)

    print(generated_text)
