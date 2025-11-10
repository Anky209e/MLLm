import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        processed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        processed = [item.strip() for item in processed if item.strip()]
        processed = [
            item if item in self.str_to_int else "<|unk|>" for item in processed
        ]
        ids = [self.str_to_int[s] for s in processed]
        return ids

    def decode(self, ids):
        # processed_text = " ".join(text)
        text = " ".join([self.int_to_str[i] for i in ids])
        processed_text = re.sub(r'\s+([,.!?"()\'])', r"\1", text)
        return processed_text


def pre_process(FILE):
    """Preporcess the data and returns token and token mappings"""

    with open(FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()
    result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    result = [item.strip() for item in result if item.strip()]
    all_words = sorted(set(result))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return all_words, vocab, vocab_size


if __name__ == "__main__":
    FILE = "Data.txt"
    all_words, vocab, vocab_size = pre_process(FILE)
    tokenizer = SimpleTokenizerV1(vocab)
    text1 = "Hello do you like some Tea?"
    text2 = "In the sunlit terraces of The place."
    text = " <|endoftext|> ".join([text1, text2])
    print(text)

    encoded = tokenizer.encode(text)
    print(encoded)

    decoded = tokenizer.decode(encoded)
    print(decoded)
