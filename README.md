# Small GPT-like Transformer

This project is an end-to-end implementation of a GPT-style decoder-only Transformer trained from scratch on a subset of the Wikipedia dataset.
The goal of the project is to move beyond toy datasets (e.g. TinyStories) and explore realistic language modeling under real-world memory and compute constraints.

The project covers the full pipeline: dataset preparation, tokenization, memory-efficient loading, model implementation, training, and metric analysis.

---

## Motivation

In a previous project, I trained a minimal Transformer on the TinyStories dataset. While useful for validating the implementation, that setup had several limitations:

* a very small and repetitive vocabulary
* limited semantic complexity
* unrealistic data loading (entire dataset kept in VRAM)

This project was designed to address those issues by:

* training on a significantly more diverse dataset (Wikipedia)
* increasing model size while fitting within available VRAM
* implementing a more realistic data pipeline (while remaining at the learning level)
* experimenting with features commonly used in modern LLMs

---

## High-Level Overview

* **Architecture:** Decoder-only Transformer (GPT-style)
* **Tokenizer:** GPT-2 BPE via `tiktoken`
* **Dataset:** Wikipedia (HuggingFace), 9 shards (~2.5GB)
* **Training length:** 50,000 steps (~450M tokens)

---

## Model Architecture

The model follows a standard GPT-like decoder-only Transformer architecture.

Key components:

* Token embedding + tied output projection
* Multi-head self-attention
* Feed-forward (MLP) blocks
* Pre-LayerNorm
* Residual connections
* RoPE for positional encoding

### Hyperparameters

The model size and hyperparameters were chosen to maximize parameter count while staying within VRAM constraints:

| Hyperparameter | Value |
|---------------|-------|
| Number of layers | 8 |
| Model dimensionality (d_model) | 512 |
| Attention heads | 8 |
| Feed-forward expansion | 4x |
| Context length | 192 |
| Positional encoding | RoPE |
| Total parameters | ~70M |

---

## Dataset & Tokenization

### Dataset

The model is trained on a subset of the Wikipedia dataset from HuggingFace:

* 9 shards
* approximately 2.5GB of raw text

Wikipedia was chosen as a more realistic alternative to synthetic or toy datasets, with a broad vocabulary and diverse topics.

### Tokenization

* Tokenizer: GPT-2 BPE (`tiktoken`)
* An explicit EOS token is inserted between documents to prevent cross-document leakage.
* All tokens are concatenated into a single continuous stream.

### Storage & Loading

The tokenized dataset is stored as a `.bin` file and accessed via memory mapping (`mmap`).

This allows:

* training on datasets larger than available VRAM/RAM
* fast random access
* minimal memory overhead

This approach replaces an earlier strategy of keeping the entire dataset in GPU memory.

---

## Training Setup

* **Optimizer:** AdamW
* **Precision:** FP32 weights with gradient scaling
* **Learning rate schedule:**

  * Linear warmup (first 5% of steps, `lr_max=8e-5`)
  * Cosine decay
* **Batch size:** 48
* **Sequence length:** 192 

The learning rate schedule was implemented to stabilize early training and allow smooth convergence over long runs. **I think LR decay was too aggressive and a longer constant LR might have been more effective.**

---

## Metrics

The following metrics were tracked throughout training:

* Raw training loss
* EMA-smoothed loss (α=0.01)
* Perplexity
* Learning rate
* Gradient norm

All metrics were visualized using `matplotlib`.
Perplexity is plotted on a log scale to better capture relative improvements over time.

EMA loss proved useful for identifying overall training trends despite noisy per-step loss values.

![plot](/plots/loss.png)

Training loss exhibits a rapid initial decrease followed by a long, stable plateau around ~4.0. This behavior is expected given the use of cosine learning rate decay and the complexity of the Wikipedia dataset. EMA-smoothed loss continues to decrease slowly, suggesting ongoing learning rather than premature convergence.

![plot](/plots/grad_norm.png)

Gradient norms remain stable throughout training, with an initial spike during early warmup followed by convergence to a narrow range (~0.8–0.9). No signs of vanishing or exploding gradients were observed, indicating stable optimization dynamics.

---

## Implemented Features & Experiments

### Successfully Implemented

* **Memory-mapped dataset loading:** scalable and realistic data pipeline
* **RoPE (Rotary Positional Embeddings):** integrated into attention mechanism
* **Learning rate scheduler:** warmup + cosine decay

### Attempted but Unsuccessful

* **Flash Attention**

Flash Attention requires FP16 activations, while this project uses FP32 weights together with gradient scaling.
Direct casting QKV tensors to FP16 is considered bad practice and was therefore avoided.

As a result, Flash Attention was not successfully integrated in this setup.

**Hypothesis of the cause**: FP32 attention weights for gradient scaling + FP32 output of LayerNorm = FP32 activations in attention -/> Flash Attention (requires FP16 activations)

---

## Results & Observations

* Training remained stable for the full 50k steps.
* Loss and perplexity decreased smoothly without divergence.
* EMA loss provided a clearer signal than raw loss.
* Gradient norms stayed within reasonable bounds, indicating stable optimization.

While no downstream evaluation was performed, the training dynamics are consistent with expected behavior for causal language models.

---

## Limitations

* No validation split or external benchmark evaluation
* No rigorous verification of RoPE correctness beyond training stability
* Limited training budget and model scale

---

## What I Learned

* Realistic datasets require memory-mapped loading; keeping data in VRAM does not scale.
* RoPE is conceptually more challenging than its implementation suggests.
* Mixed-precision training involves subtle interactions between weights, activations, and optimizers.
* Monitoring gradient norms is critical for diagnosing training stability and trend.
