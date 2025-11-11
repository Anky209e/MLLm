import tiktoken

text = (
    "Hello would you like to drink some tea? <|endoftext|> In the sunlit terraces "
    "Of someunkownplace"
)

tokenizer = tiktoken.get_encoding("gpt2")

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(ids)

decoded_text = tokenizer.decode(ids)

print(decoded_text)
