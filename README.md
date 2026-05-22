# GPT-Style Transformer: From Scratch to Fine-Tuning

A comprehensive, hands-on implementation of a GPT-style transformer model built entirely from scratch using PyTorch — covering tokenization, architecture design, pretraining, and fine-tuning for both **classification** and **instruction-following** tasks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Key Concepts Covered](#key-concepts-covered)
- [Model Configurations](#model-configurations)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project walks through the complete lifecycle of a large language model (LLM) — from raw text tokenization to fine-tuning a pretrained GPT-2 model on downstream tasks. It is structured as two detailed Jupyter notebooks that serve both as educational resources and practical implementation guides.

The work covers:

- Building a custom tokenizer and Byte Pair Encoding (BPE) pipeline
- Implementing self-attention and multi-head attention from scratch
- Constructing the full GPT model architecture in PyTorch
- Training the model on raw text with a next-token prediction objective
- Loading pretrained OpenAI GPT-2 weights
- Fine-tuning GPT-2 for **spam classification** (classification head)
- Fine-tuning GPT-2 for **instruction following** using the Alpaca format (supervised fine-tuning / SFT)
- Evaluating fine-tuned models using Llama 3 via Ollama

---

## Project Structure

```
GPT-style_transformer_for_classification_and_prediction_task/
│
├── Architecture_Classification_finetuning.ipynb   # GPT architecture + spam classification fine-tuning
├── LLM_finetuning_training.ipynb                  # LLM training loop + instruction fine-tuning (SFT)
├── the-verdict.txt                                # Sample text corpus used for tokenization demos
└── README.md
```

---

## Notebooks

### 1. `Architecture_Classification_finetuning.ipynb`

Covers the full pipeline from raw text to a fine-tuned classification model:

| Section | Description |
|---|---|
| Tokenization | Custom tokenizer V1/V2, BPE with `tiktoken` |
| Input-Target Pairs | Sliding window dataset construction |
| Token Embeddings | Token + positional embeddings |
| Self-Attention | Scaled dot-product attention from scratch |
| Multi-Head Attention | Parallel attention heads with causal masking |
| GPT Architecture | Full `GPTModel` class with transformer blocks |
| Text Generation | `generate_text_simple` with temperature & top-k sampling |
| Loss & Evaluation | Cross-entropy loss, perplexity |
| Classification Fine-Tuning | GPT-2 fine-tuned on SMS Spam Collection dataset |

---

### 2. `LLM_finetuning_training.ipynb`

Extends the architecture notebook with a full training loop and instruction fine-tuning:

| Section | Description |
|---|---|
| Training Loop | LLM pretraining with train/val loss tracking |
| Decoding Strategies | Temperature scaling, top-k sampling |
| Pretrained Weights | Loading OpenAI GPT-2 weights via `gpt_download3` |
| Instruction Fine-Tuning | Alpaca-format SFT on an instruction dataset |
| Dataset Preparation | Train/validation/test splits with custom DataLoader |
| Model Evaluation | Automated scoring using Llama 3 via Ollama REST API |
| Model Saving | Saving and reloading fine-tuned `.pth` checkpoints |

---

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended; CPU works but is slow)
- [Ollama](https://ollama.com/) (required only for automated evaluation in `LLM_finetuning_training.ipynb`)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/GPT-style-transformer.git
cd GPT-style-transformer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install tiktoken
pip install pandas numpy matplotlib tqdm requests
```

> **Note:** For GPU support, install the appropriate PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

### Run the notebooks

```bash
jupyter notebook
```

Open and run either notebook in order:

1. Start with **`Architecture_Classification_finetuning.ipynb`** to understand the architecture and classification fine-tuning.
2. Then proceed to **`LLM_finetuning_training.ipynb`** for the full training loop and instruction fine-tuning.

### For Automated Evaluation (Ollama)

Install Ollama and pull the Llama 3 model before running the evaluation cells:

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3
ollama serve
```

---

## Key Concepts Covered

- **Tokenization**: Simple regex-based tokenizers, vocabulary construction, and Byte Pair Encoding (BPE) using `tiktoken` (GPT-2, GPT-3, GPT-4 encodings)
- **Attention Mechanism**: Scaled dot-product self-attention, causal (masked) attention, multi-head attention
- **Transformer Block**: Layer normalization, feed-forward network, residual connections, dropout
- **GPT Architecture**: Full autoregressive language model built from first principles
- **Pretraining Objective**: Next-token prediction with cross-entropy loss
- **Decoding**: Greedy decoding, temperature scaling, top-k sampling
- **Transfer Learning**: Loading pretrained GPT-2 weights and adapting for downstream tasks
- **Classification Head**: Replacing the LM head with a binary classification output layer
- **Instruction Fine-Tuning**: Supervised fine-tuning using Alpaca-format prompt templates
- **Evaluation**: Accuracy metrics for classification; LLM-as-judge scoring for instruction following

---

## Model Configurations

The notebooks support all four GPT-2 model sizes:

| Model | Parameters | Embedding Dim | Layers | Attention Heads |
|---|---|---|---|---|
| GPT-2 Small | 124M | 768 | 12 | 12 |
| GPT-2 Medium | 355M | 1024 | 24 | 16 |
| GPT-2 Large | 774M | 1280 | 36 | 20 |
| GPT-2 XL | 1558M | 1600 | 48 | 25 |

Default configuration used across notebooks:

```python
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
```

---

## Datasets

| Task | Dataset | Source |
|---|---|---|
| Text corpus (tokenization demo) | *The Verdict* (short story) | Included as `the-verdict.txt` |
| Spam Classification | SMS Spam Collection | [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — auto-downloaded |
| Instruction Fine-Tuning | Alpaca-style instruction data | Downloaded automatically in notebook |

The spam dataset is automatically downloaded, balanced (ham/spam 50-50), and split into 70% train / 10% validation / 20% test.

---

## Results

### Spam Classification (GPT-2 Small fine-tuned)
- Task: Binary classification (spam vs. ham)
- Approach: Frozen GPT-2 backbone; only the last transformer block, final norm, and a new linear classification head are trained
- Dataset: Balanced SMS Spam Collection (~1,600 samples per class)

### Instruction Following (GPT-2 Medium fine-tuned)
- Task: Instruction-following in Alpaca prompt format
- Approach: Supervised fine-tuning (SFT) on a curated instruction dataset
- Evaluation: Automated response scoring using Llama 3 (0–100 scale) via Ollama REST API
- Saved checkpoint: `gpt2-medium355M-sft.pth`

---

## Contributing

Contributions, issues, and feature requests are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- Architecture and training approach inspired by Sebastian Raschka's *"Build a Large Language Model (From Scratch)"*
- Pretrained weights loaded from [OpenAI's GPT-2](https://github.com/openai/gpt-2)
- BPE tokenization via [tiktoken](https://github.com/openai/tiktoken)
- Instruction evaluation using [Ollama](https://ollama.com/) + Meta's Llama 3
