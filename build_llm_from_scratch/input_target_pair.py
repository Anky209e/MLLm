import tiktoken

FILE = "Data.txt"

with open(FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

encoded_text = tokenizer.encode(raw_text)

encoded_sample = encoded_text[50:]

CONTEXT_LENGTH = 4

for i in range(1, CONTEXT_LENGTH + 1):
    context = encoded_sample[:i]
    target = encoded_sample[i]

    print(f"{tokenizer.decode(context)} -----> {tokenizer.decode([target])}")
