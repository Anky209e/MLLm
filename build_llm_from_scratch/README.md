# 🧠 Building LLMs from Scratch: A Deep Dive

This repository contains a step-by-step implementation of a GPT-style Large Language Model (LLM). These notes serve as a guide to understanding the architecture from the basic attention mechanism to the full Transformer block.

---

## 🗺️ Roadmap & Architecture Overview

The journey from raw text to a model that "understands" context follows this path:
`Raw Text` → `Tokens` → `Embeddings` → `Transformer Blocks` → `Logits` → `Probabilities`.

### 1. Tokenization & Input Preparation
Before the model can process text, it must be converted into numbers.
- **Byte-Pair Encoding (BPE):** A subword tokenization method that balances vocabulary size and sequence length.
- **Files:** `byte_pair_tik_token.py`, `input_target_pair.py`

### 2. The Embedding Layer
We represent tokens in a continuous multi-dimensional space.
- **Token Embeddings:** Mapping each token ID to a vector of size `d_model`.
- **Positional Embeddings:** Since Transformers process all tokens in parallel, we must "inject" order information. We add a unique vector to each token based on its position.
- **Formula:** $X_{final} = X_{token} + X_{position}$
- **Files:** `embedding_lookup.py`, `positional_embeddings.py`, `gpt.py`

---

## ⚡ The Attention Mechanism: The "Brain" of the LLM

Attention allows the model to focus on relevant parts of the input sequence when processing a specific token.

### A. Simple Attention (The Intuition)
The core idea is **Similarity**. We calculate how much one token "attends" to another using the Dot Product.
- **Higher Dot Product** = Higher Similarity = More Attention.
- **File:** `simple_attention.py`

### B. Self-Attention ($Q, K, V$)
We introduce trainable weights to allow the model to learn *what* to look for.
1. **Query ($Q$):** "What am I looking for?"
2. **Key ($K$):** "What information do I contain?"
3. **Value ($V$):** "What information should I pass along?"

**The Math:**
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Scaling by $\sqrt{d_k}$:** Prevents the dot product from growing too large, which would push the softmax into regions with tiny gradients (vanishing gradient problem).
- **File:** `self_attention.py`

### C. Causal Attention (The "Look-Ahead" Mask)
In GPT (Generative Pre-trained Transformer), a token should only see its predecessors, not the future.
- **Implementation:** We apply a **Mask** to the attention scores.
- **Why $-\infty$?** When we apply `softmax`, $e^{-\infty} = 0$, effectively "turning off" the future tokens.
- **File:** `causal_attention.py`

### D. Multi-Head Attention (MHA)
Instead of one "viewpoint," we use multiple "heads" to attend to different types of information simultaneously (e.g., one head for grammar, one for factual relationships).
- **Process:** Split $d_{out}$ into $h$ heads → Process in parallel → Concatenate → Project back.
- **File:** `multi_head_attention.py`

---

## 🧱 The Transformer Block
The Transformer block combines MHA with other essential components to ensure stable training.

```text
Input ──► LayerNorm ──► Multi-Head Attention ──► Dropout ──► (+) ──┐
  │                                                          ▲     │ (Residual Connection)
  └──────────────────────────────────────────────────────────┘     │
                                                                   │
┌──────────────────────────────────────────────────────────────────┘
│
└───► LayerNorm ──► Feed Forward (GELU) ──► Dropout ──► (+) ──► Output
  │                                                      ▲      (Residual Connection)
  └──────────────────────────────────────────────────────┘
```

### Key Components:
- **Layer Normalization:** Normalizes the inputs to a layer to have mean 0 and variance 1. This stabilizes training and prevents "Internal Covariate Shift." (`layer_normalisation.py`)
- **GELU (Gaussian Error Linear Unit):** A smoother version of ReLU that allows small negative values, helping gradients flow better. (`gelu.py`)
- **Residual (Skip) Connections:** Adding the input of a block to its output ($x + f(x)$). This allows gradients to flow through the network without disappearing in deep architectures.
- **Feed Forward Network (FFN):** Two linear layers with a GELU activation in between, expanding the dimension usually by 4x. (`feed_forward_network.py`)
- **File:** `transformer.py`

---

## 🏗️ The Full GPT Model
Assembling everything into the final architecture.
1. **Embedding Layer** (Token + Positional)
2. **Stack of N Transformer Blocks**
3. **Final Layer Norm**
4. **Output Linear Layer:** Projects the $d_{model}$ back to $V$ (vocab size) to get "Logits."
- **File:** `gpt.py`

---

## 🚀 Training & Inference

### Training Logic
- **Objective:** Predict the next token.
- **Loss Function:** Cross-Entropy Loss between predicted logits and target token IDs.
- **Files:** `main.py`, `finetune.py`, `loss-plot.pdf`

### Inference Logic
1. Start with a prompt.
2. Get the model's prediction for the last token.
3. Append the predicted token to the input.
4. Repeat until a limit is reached or an `<endoftext>` token is generated.
- **File:** `inference.py`

---

## 📊 Summary Cheat Sheet

| Component | Purpose | Key File |
| :--- | :--- | :--- |
| **BPE** | Efficient Tokenization | `byte_pair_tik_token.py` |
| **Scaling** ($\sqrt{d_k}$) | Gradient Stability | `self_attention.py` |
| **Masking** | Prevents Data Leakage | `causal_attention.py` |
| **Multi-Head** | Parallel Perspectives | `multi_head_attention.py` |
| **Residuals** | Trains Deep Networks | `transformer.py` |
| **Logits** | Raw prediction scores | `gpt.py` |

```
