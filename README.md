# 🧠 MLLm: Machine Learning & Large Language Models from Scratch

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9.0+](https://img.shields.io/badge/pytorch-2.9.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MLLm** is an educational repository dedicated to demystifying the architecture and training of Large Language Models (LLMs). This project guides you through the complete journey of building a GPT-style transformer from the ground up, fine-tuning it for instruction-following, and adapting it for specialized classification tasks.

---

## 🗺️ Learning Path

This repository is organized into three progressive modules. We recommend following them in this order:

1.  **[Build from Scratch](./build_llm_from_scratch/)**: Understand the core architecture. Implement tokenization, embeddings, multi-head attention, and transformer blocks.
2.  **[Instruction Fine-tuning](./instruction_finetuning/)**: Learn how to take a pre-trained model and teach it to follow user instructions using the Alpaca format.
3.  **[Spam Classification](./llm_spam_classification/)**: Master transfer learning. Learn how to freeze model layers and adapt a language model for binary classification.

---

## 🏗️ Core Architecture

The project implements a **Decoder-only Transformer** (GPT-style) with the following components:

- **Tokenization**: Byte-Pair Encoding (BPE) using OpenAI's `tiktoken`.
- **Embeddings**: Combined Token and Positional embeddings.
- **Attention Mechanism**: Scaled dot-product self-attention with causal masking (look-ahead mask).
- **Multi-Head Attention**: Parallel attention heads for capturing diverse context.
- **Transformer Block**: Integrated Layer Normalization, GELU activations, and Residual (Skip) connections.
- **Output Head**: Linear layer projecting to vocabulary size (for generation) or class size (for classification).

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.12 or higher
- **GPU**: CUDA-compatible GPU recommended for training (but not required for inference)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MLLm

# Using uv (recommended)
uv sync

# OR using standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## 📦 Modules Overview

### 1. Building LLMs from Scratch
Located in `/build_llm_from_scratch`. This module contains the foundational code for the transformer architecture.

- **Training**: Train a small GPT model on Project Gutenberg books.
- **Inference**: Generate text using greedy or top-k sampling.
- **Key Files**: `gpt.py`, `transformer.py`, `multi_head_attention.py`, `causal_attention.py`.

```bash
cd build_llm_from_scratch
python main.py      # Start training
python inference.py # Interactive generation
```

### 2. Instruction Fine-tuning
Located in `/instruction_finetuning`. Adapts a pre-trained GPT-2 (124M) model to follow instructions.

- **Dataset**: Uses the Alpaca-style instruction dataset (Instruction, Input, Response).
- **Technique**: Supervised Fine-Tuning (SFT) on the full model.
- **Key Files**: `dataset.py`, `utils.py` (formatting & collate functions).

```bash
cd instruction_finetuning
python download_instruction_dataset.py
python main.py # Start fine-tuning
```

### 3. LLM Spam Classification
Located in `/llm_spam_classification`. Demonstrates how to repurpose an LLM for specialized tasks.

- **Transfer Learning**: Loads pre-trained GPT-2 weights and replaces the language modeling head with a classification head.
- **Selective Unfreezing**: Freezes earlier layers to preserve general language knowledge while training only the final transformer block and head.
- **Key Files**: `finetune.py`, `utils.py` (accuracy & loss monitoring).

```bash
cd llm_spam_classification
python download_sms_data.py
python main.py # Start classification training
```

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, TensorFlow (utilities)
- **NLP**: Tiktoken, Scikit-learn
- **Visualization**: Matplotlib, Plotly, TensorBoard
- **Data Handling**: Pandas, NumPy
- **Productivity**: TQDM (progress bars), Jupyter

---

## 📖 Theoretical Concepts Covered

- **Causal Masking**: Why $e^{-\infty} = 0$ is the key to generative models.
- **Residual Connections**: Solving the vanishing gradient problem in deep networks.
- **Layer Normalization**: Stabilizing training and preventing internal covariate shift.
- **Transfer Learning**: The power of selective unfreezing and head replacement.

---

## 🤝 Contributing

Contributions are welcome! Whether it's adding a new model architecture (like RoPE embeddings), improving the training loop, or adding more datasets, feel free to open a PR.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
