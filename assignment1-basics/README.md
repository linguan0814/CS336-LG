# Pretraining Basics

[中文](README.zh-CN.md) | English

The foundational module for the repository: a small causal Transformer language model and its training pipeline implemented without a high-level model framework.

## Included components

- GPT-style regex pre-tokenization and byte-pair encoding (BPE) training, serialization, and encode/decode utilities.
- Transformer building blocks: embedding, Linear, RMSNorm, SwiGLU, RoPE, scaled dot-product attention, multi-head attention, and pre-norm Transformer blocks.
- Language-model training utilities for memmap data loading, AdamW, learning-rate scheduling, gradient clipping, validation, checkpointing, and sampling.
- Tests for tokenization, BPE training, serialization, data handling, optimizer behavior, modules, and the Transformer LM.

The model-backed test fixture is retained locally but excluded from the public
release by the root `.gitignore`; tokenizer and component tests do not depend
on that weight file.

## Design at a glance

```text
raw text → BPE training → tokenizer artifacts → tokenized memmap
                                             ↓
                              Transformer LM → train / validate → generate
```

The implementation keeps the model and training concerns separate: `model/` owns neural-network building blocks, `trainer/` owns optimization and data utilities, and the top-level modules expose tokenization, training, and generation entry points. [Design notes](docs/README.md) map those concepts to the concrete files.

```text
assignment1-basics/
├── cs336_basics/             # Tokenizer, model, training, and generation code
├── tests/                    # Unit tests and small fixtures
├── docs/                     # Design notes and the selected training curve
├── scripts/                  # Report/plot helpers
├── pyproject.toml
└── uv.lock
```

## Run

```bash
uv run pytest
```

The data-preparation, tokenizer-training, model-training, and generation entry points live under `cs336_basics/`. They accept local input/output paths; datasets, tokenized corpora, checkpoints, and raw W&B runs are intentionally not committed. The selected training artifact is documented in `docs/` without shipping training state.

## Recorded evidence

The committed validation-loss curve is a lightweight record of a completed run. It should be read as a training-health trace, not as a comparison to an external model or benchmark.

![Validation loss over a completed A1 run](docs/assets/a1_validation_loss.png)
