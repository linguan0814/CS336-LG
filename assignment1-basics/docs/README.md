# A1 design notes

[中文](README.zh-CN.md) | English

## Objective

Build a compact causal language-model pipeline from raw text to generation while exposing the components normally hidden behind a framework.

## Architecture

| Stage | Public code | Responsibility |
| --- | --- | --- |
| Tokenization | `train_bpe.py`, `tokenizer.py`, `scripts/train_tokenizer.py` | GPT-style pre-tokenization, BPE merge learning, serialization, encode/decode |
| Data | `scripts/tokenize_dataset.py`, `trainer/data_loading.py` | Convert text to token-id memmaps and sample training batches |
| Model | `model/modules.py`, `model/transformer.py` | Embedding, RMSNorm, SwiGLU, RoPE, causal attention, Transformer LM |
| Optimization | `trainer/AdamW.py`, `trainer/utils.py` | AdamW, schedules, loss, gradient clipping |
| Training and inference | `train.py`, `generate.py`, `check_pointing.py` | Train/validate, checkpoint lifecycle, temperature/top-p generation |

## Verification and evidence

`tests/` covers the key interfaces and behavior. `assets/a1_validation_loss.png` is the single selected training curve retained for public review. Additional rendered curves, source corpora, tokenized data, checkpoints, and raw tracking files are intentionally local-only; they are not required to inspect the implementation or run unit tests.

## Reproduction boundary

Run `uv run pytest` for the checked-in test suite. Full training additionally requires a locally prepared corpus and sufficient compute. The public project does not claim that a reviewer can reproduce the original training run without independently obtaining those inputs.
