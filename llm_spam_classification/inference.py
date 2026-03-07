from utils import classify_review
from gpt import GPTModel
import torch
import tiktoken

if __name__ == "__main__":
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "embedding_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.15,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }
    WEIGHT_PATH = "training_end_weight.pth"

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(BASE_CONFIG)
    model.output_layer = torch.nn.Linear(
        in_features=BASE_CONFIG["embedding_dim"], out_features=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.to(device)

    text_1 = "Congratulations! Yow won the grand lottery of 1000$"

    probs = classify_review(
        text_1, model=model, tokenizer=tokenizer, device=device, max_length=92
    )
    preds = torch.argmax(probs, dim=-1).item()
    classes = ["HAM", "SPAM"]
    print(f"Probs:{probs}\n Prediction:{classes[preds]}")
