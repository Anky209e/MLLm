# Instruction Fine-tuning

Fine-tuning GPT-2 (124M) on instruction-following data to make the model respond to user instructions.

## Overview

Uses the Alpaca format for instruction data:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Training script - loads instruction data, fine-tunes GPT-2 |
| `dataset.py` | InstructionDataset - encodes instruction+response pairs |
| `finetune.py` | load_weights_into_gpt() - transfers pretrained GPT-2 weights |
| `utils.py` | Training utilities, Alpaca formatting, collate function, generation |
| `download_gpt.py` | Downloads GPT-2 weights from OpenAI |
| `download_instruction_dataset.py` | Downloads instruction training data |

## Architecture

Same GPT architecture as `build_llm_from_scratch`:
- `gpt.py` - GPTModel with token/position embeddings
- `transformer.py` - Transformer block (attention + FFN)
- `multi_head_attention.py` - Multi-head causal attention
- `feed_forward_network.py` - Feed-forward network
- `layer_normalisation.py` - Layer normalization
- `gelu.py` - GELU activation

## Training

```bash
python main.py
```

- Downloads instruction data from GitHub
- Splits: 85% train, 10% test, 5% validation
- Loads pretrained GPT-2 124M weights
- Fine-tunes with lower learning rate (5e-5)
- Saves checkpoints every epoch

## Key Functions

### utils.py

```python
format_to_alpaca(entry)  # Formats instruction to Alpaca prompt
collate(batch, device, allowed_max_length)  # Pads sequences, creates targets
generate(model, idx, max_new_tokens, context_size, temperature, top_k, eos_id)
train_model_simple(...)  # Training loop with evaluation
```

### finetune.py

```python
load_weights_into_gpt(gpt_model, params)  # Transfers GPT-2 weights to custom model
```

## Configuration

```python
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.15,
    "qkv_bias": True,  # GPT-2 uses bias
}
```

## Data Format

Expected JSON format:
```json
{
  "instruction": "Explain gravity",
  "input": "",
  "output": "Gravity is a force..."
}
```

## Notes

- Uses tiktoken GPT-2 tokenizer
- EOS token (50256) stops generation
- Padding handled with special tokens
- Target labels ignore padding tokens
