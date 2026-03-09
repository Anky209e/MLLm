# LLM Spam Classification

Fine-tuning GPT-2 (124M) for SMS spam classification using transfer learning.

## Overview

Uses a pretrained GPT-2 model and adapts it for binary classification (spam vs. ham).

## Files

| File | Description |
|------|-------------|
| `main.py` | Training script - loads SMS data, fine-tunes for classification |
| `dataset.py` | SpamDataset - loads CSV, tokenizes SMS messages |
| `finetune.py` | load_weights_into_gpt() - transfers pretrained GPT-2 weights |
| `utils.py` | Training loop, accuracy calculation, classification, plotting |
| `download_gpt.py` | Downloads GPT-2 weights from OpenAI |

## Architecture

Same GPT architecture:
- `gpt.py` - GPTModel
- `transformer.py` - Transformer block
- `multi_head_attention.py` - Multi-head causal attention
- `feed_forward_network.py` - Feed-forward network
- `layer_normalisation.py` - Layer normalization
- `gelu.py` - GELU activation

## Training Strategy

1. Load pretrained GPT-2 124M weights
2. **Freeze** all model parameters
3. Replace output layer: `vocab_size → 2` (spam/ham)
4. **Unfreeze** only:
   - Last transformer block
   - Final layer normalization
5. Train with lower learning rate (5e-5)

## Why Freeze + Selective Unfreeze?

- **Freeze**: Pretrained weights already encode language understanding
- **Unfreeze last layer**: Allows model to adapt task-specific features
- **Benefits**: Faster training, prevents catastrophic forgetting, less GPU memory

## Data Format

CSV with columns:
- `Text` - SMS message content
- `Label` - 0 (ham) or 1 (spam)

Expected files in `sms_data/`:
- `train.csv`
- `validation.csv`
- `test.csv`

## Usage

```bash
python main.py
```

Training:
- 5 epochs
- Batch size 8
- AdamW optimizer (lr=5e-5, weight_decay=0.1)
- Saves best model by accuracy

## Key Functions

### utils.py

```python
loader_accuracy(data_loader, model, device)  # Calculate accuracy
batch_loss(input_batch, target_batch, model, device)  # Cross-entropy loss
train_classifier(model, train_loader, val_loader, optimizer, ...)  # Training loop
classify_review(text, model, tokenizer, device)  # Single prediction
plot_values(...)  # Plot training curves
```

### dataset.py

```python
SpamDataset(csv_file, tokenizer, max_length)  # Loads SMS data
# Pads all sequences to max_length
# Returns (input_ids, label) tensors
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
    "qkv_bias": True,
}

num_classes = 2  # spam, ham
```

## Inference Example

```python
from utils import classify_review

probs = classify_review(
    "Free win! Click here now!",
    model, tokenizer, device
)
# probs[0][0] = probability of ham
# probs[0][1] = probability of spam
```

## Notes

- Uses last token logits for classification (standard practice for BERT/GPT)
- Tokenizer: tiktoken GPT-2
- Padding token: 50256 (end-of-text)
- Training saves checkpoints based on accuracy
