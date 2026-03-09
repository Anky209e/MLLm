import torch
import tiktoken

from download_gpt import download_and_load_gpt2
from finetune import load_weights_into_gpt
from gpt import GPTModel
from utils import generate, text_to_token_ids, token_ids_to_text


FIXED_INSTRUCTION = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain the concept of"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

tokenizer = tiktoken.get_encoding("gpt2")

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

settings, params = download_and_load_gpt2("124M", "downloaded_models")
model = GPTModel(cfg=BASE_CONFIG)
load_weights_into_gpt(model, params)

state_dict = torch.load("final_checkpoint.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()
print("Model loaded!")


def infer(user_input: str, max_new_tokens: int = 256) -> str:
    prompt = f"{FIXED_INSTRUCTION} {user_input}"

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer, device=device),
        max_new_tokens=max_new_tokens,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    response = generated_text[len(prompt) :].strip()
    return response


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter topic (or 'q' to quit): ")
        if user_input.lower() == "q":
            break

        response = infer(user_input)
        print(f"\n{response}")
