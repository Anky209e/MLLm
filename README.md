# MLLm - Machine Learning & Large Language Models from Scratch

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive educational project for implementing machine learning and large language models from scratch. This project focuses on building GPT-style transformer models and applying them to real-world tasks like text generation and classification.

## 🎯 Project Overview

**MLLM** is designed to help you understand the inner workings of modern language models by implementing them from the ground up. The project covers:

- **Complete Transformer Architecture**: Multi-head attention, layer normalization, feed-forward networks
- **GPT-Style Language Models**: From basic concepts to full implementation
- **Training Pipeline**: Data loading, training loops, checkpointing, and visualization
- **Transfer Learning**: Fine-tuning pre-trained models for specific tasks
- **Real-World Applications**: Text generation and SMS spam classification

## 🏗️ Project Structure

```
MLLm/
├── build_llm_from_scratch/          # Main LLM implementation
│   ├── gpt.py                       # GPT model architecture
│   ├── transformer.py               # Transformer block
│   ├── multi_head_attention.py      # Multi-head attention
│   ├── layer_normalisation.py       # Layer normalization
│   ├── feed_forward_network.py      # Feed-forward network
│   ├── main.py                      # Training script
│   ├── inference.py                 # Text generation
│   └── README.md                    # Detailed concept explanations
│
└── llm_spam_classification/         # Fine-tuning application
    ├── main.py                      # Spam classification training
    ├── finetune.py                  # Fine-tuning utilities
    └── dataset.py                   # SMS dataset handling
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (optional but recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MLLm

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Training Your First Model

```bash
# Train a GPT model from scratch
cd build_llm_from_scratch
python main.py

# Generate text with your trained model
python inference.py

# Fine-tune for spam classification
cd ../llm_spam_classification
python main.py
```

## 📚 Key Features

### 🧠 Complete Transformer Implementation
- **Multi-Head Attention**: Full implementation with causal masking
- **Layer Normalization**: Prevents vanishing/exploding gradients  
- **GELU Activation**: Modern activation function for better performance
- **Positional Embeddings**: Captures sequential information
- **Residual Connections**: Standard transformer architecture patterns

### 🎯 Training Pipeline
- **Custom Dataset**: Efficient text data handling
- **Batch Processing**: Configurable batch sizes and context lengths
- **Training Loop**: Comprehensive training with validation and checkpointing
- **TensorBoard Integration**: Real-time training visualization
- **Loss Plotting**: Automatic generation of training metrics

### 🔤 Text Generation
- **Multiple Sampling Strategies**: Greedy, temperature-based, top-k sampling
- **Context-Aware Generation**: Maintains conversation context
- **Configurable Parameters**: Fine-tune generation behavior

### 🔄 Transfer Learning
- **Pre-trained Weight Loading**: Download and load GPT-2 weights
- **Fine-tuning**: Adapt models for specific tasks
- **Layer Freezing**: Selective training of model components
- **Classification Heads**: Replace language modeling for classification tasks

## 🛠️ Technology Stack

- **Core**: Python 3.12+, PyTorch 2.9.0+, TensorFlow 2.20.0+
- **Tokenization**: tiktoken 0.12.0+ (OpenAI's BPE tokenizer)
- **Data**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Plotly, TensorBoard
- **Development**: Jupyter, tqdm

## 📖 Learning Resources

The `build_llm_from_scratch/README.md` contains detailed explanations of:

- **Tokenization**: Word-based vs. BPE tokenization
- **Embeddings**: Word and positional embeddings
- **Attention**: Simple attention, self-attention, causal attention, multi-head attention
- **Transformers**: Complete architecture breakdown
- **Training**: Loss functions, optimization, and evaluation

## 🎯 Example Use Cases

### 1. Language Model Training
```bash
cd build_llm_from_scratch
python main.py
```
Trains a GPT-style model on Project Gutenberg books with automatic checkpointing and visualization.

### 2. Text Generation
```bash
python inference.py
```
Interactive text generation with multiple sampling strategies and configurable parameters.

### 3. Spam Classification
```bash
cd llm_spam_classification
python main.py
```
Fine-tunes a pre-trained GPT model for SMS spam classification with high accuracy.

## 🔧 Configuration

Models use configuration dictionaries for easy experimentation:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout_rate": 0.1,
    "qkv_bias": False
}
```

## 📊 Training Features

- **Automatic Checkpointing**: Save model progress during training
- **Validation Monitoring**: Track training and validation metrics
- **TensorBoard Logging**: Real-time visualization of training progress
- **GPU Support**: Automatic CUDA detection and utilization
- **Flexible Data Loading**: Configurable batch sizes and data splits

## 🧪 Model Capabilities

The implemented models support:

- **Text Generation**: Creative writing, completion, and dialogue
- **Classification**: Binary and multi-class classification tasks
- **Fine-tuning**: Adaptation to specific domains and tasks
- **Transfer Learning**: Leverage pre-trained knowledge
- **Custom Training**: Train on your own datasets

## 📈 Performance

- **Training Speed**: Optimized for GPU acceleration
- **Memory Efficiency**: Careful batch management and gradient checkpointing
- **Scalability**: Supports various model sizes from small to large
- **Accuracy**: Competitive performance on classification tasks

## 🤝 Contributing

This project is primarily educational, but contributions are welcome! Areas for improvement:

- Additional model architectures
- More training datasets
- Enhanced visualization tools
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the original "Attention Is All You Need" transformer paper
- Inspired by OpenAI's GPT architecture
- Uses Project Gutenberg for training data
- Incorporates modern deep learning best practices

## 🔗 Related Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [OpenAI GPT Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_textual.pdf)
- [tiktoken Documentation](https://github.com/openai/tiktoken)

---

**Note**: This project is primarily designed for educational purposes. For production use cases, consider established libraries like Hugging Face Transformers.