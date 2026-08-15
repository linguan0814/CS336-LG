# Alignment Training

[中文](README.zh-CN.md) | English

Post-training implementations and experiments for mathematical reasoning. The module combines supervised fine-tuning (SFT), Expert Iteration (EI), and GRPO-style policy optimization with rule-based format and answer rewards.

## Included components

- Prompt construction, response-only SFT loss, data preparation, rollout, and evaluation utilities.
- Rule-based rewards that check structured `<think>...</think><answer>...</answer>` responses and mathematical-answer equivalence.
- SFT, EI, and GRPO training entry points in `cs336_alignment/`.
- Tests and small local model/tokenizer fixtures for core data and evaluation behavior.
- Curated experiment figures and an anonymized GRPO result summary in `docs/`.

The tiny GPT-2 model weights used by one DPO fixture test remain local and are
excluded from the public release; tokenizer fixtures and the source tests are
kept for reference.

## Design at a glance

```text
prompt + verifier → rollout → format / answer rewards
                         ├──→ SFT on response tokens
                         ├──→ Expert Iteration from verified samples
                         └──→ GRPO-style relative-reward updates
```

The code separates reward/evaluation utilities from the training loops so that SFT, EI, and GRPO share the same task contract. [Design notes](docs/README.md) describe the experiment scope and how to interpret the retained curves.

## Selected GRPO evidence

The public result summary focuses on two controlled Math12K ablations rather than comparing unrelated SFT/EI and GRPO runs:

- Disabling group standard-deviation normalization: terminal accuracy **48.4% → 61.4%**, with longer responses.
- DAPO-style length normalization: terminal accuracy **39.0% → 48.2%**, while reducing overall response length.

See [the selected result table](docs/results.md) for the matched settings, correct/incorrect response lengths, and limits on interpretation.

```text
assignment5-alignment/
├── cs336_alignment/          # Training loops, rewards, data, prompts, and evaluation
├── scripts/                  # Data preparation, evaluation, and reporting utilities
├── tests/                    # Unit tests and compact fixtures
├── docs/                     # Design notes, selected results, and four evidence figures
├── pyproject.toml
└── uv.lock
```

## Run

```bash
uv run pytest

# Inspect available training options; full runs need local data/model paths and CUDA.
uv run python -m cs336_alignment.train_sft_step --help
uv run python -m cs336_alignment.train_ei_step --help
uv run python -m cs336_alignment.train_grpo --help
```

Full training requires compatible CUDA hardware and local datasets/model weights. They, along with checkpoints, raw W&B exports, and runtime logs, are excluded from version control. The retained documentation figures record existing experiments and should not be treated as newly rerun results.
