# CS336-LG

[中文](README.zh-CN.md) | English

### From-scratch LLM training, systems optimization, and alignment

An engineering portfolio built around the LLM development stack: implementing a tokenizer and Transformer language model from first principles, studying attention and decoding efficiency, and training aligned reasoning policies with SFT, Expert Iteration, and GRPO.

The repository is organized as independently runnable modules. Assignment 1, Assignment 2, and Assignment 5 are the primary project artifacts; the scaling and data-pipeline modules remain as supporting work.

## Highlights

- **Pretraining fundamentals:** byte-level BPE, Transformer components, language-model training, checkpointing, and generation without a high-level model framework.
- **Systems focus:** tiled PyTorch/Triton attention, KV-cache decoding, and instructional DDP/FSDP/optimizer-sharding implementations.
- **Alignment workflows:** response-only SFT, rule-based math rewards, Expert Iteration, and GRPO-style policy optimization.
- **Evidence over artifacts:** tests, reproducible entry points, curated plots, and benchmark tables are committed; datasets, weights, checkpoints, and raw tracking logs stay local.

## Repository map

```text
.
├── assignment1-basics/       # BPE, Transformer LM, training loop, tests, and curves
├── assignment2-systems/      # Attention kernels, KV cache, distributed systems, benchmarks
├── assignment3-scaling/      # Supporting scaling-law study
├── assignment4-data/         # Supporting data-filtering pipeline
└── assignment5-alignment/    # SFT / EI / GRPO training, evaluation, and experiment figures
```

Each module has its own `pyproject.toml` and `uv.lock`; install and run it from that module's directory.

## What is implemented

| Area | Main implementations | Documentation |
| --- | --- | --- |
| **A1 — Pretraining** | BPE training, tokenizer, RMSNorm/SwiGLU/RoPE, causal attention, Transformer LM, AdamW, training and generation | [module README](assignment1-basics/README.md) · [design notes](assignment1-basics/docs/README.md) |
| **A2 — Systems** | Tiled PyTorch attention, Triton forward kernel, KV cache, DDP, FSDP, optimizer-state sharding | [module README](assignment2-systems/README.md) · [design notes](assignment2-systems/docs/README.md) |
| **A5 — Alignment** | SFT, EI, GRPO, rule-based format/answer rewards, rollout and evaluation utilities | [module README](assignment5-alignment/README.md) · [design notes](assignment5-alignment/docs/README.md) · [selected results](assignment5-alignment/docs/results.md) |

## Selected evidence

These are recorded results from the completed experiments—not newly rerun claims. Conditions and limitations are documented next to each module.

### A1 — Transformer training converges on the held-out split

The completed 30k-step run shows the validation loss falling from roughly 6.8 to the mid-5s. The figure is intentionally kept as a compact training-health signal; the dataset, tokenized corpus, and checkpoint are local-only.

<p align="center">
  <img src="assignment1-basics/docs/assets/a1_validation_loss.png" alt="A1 Transformer validation loss over training" width="900">
</p>

### A2 — compiler baseline and memory boundary

For non-causal float32 attention at batch size 8, `torch.compile` improved the recorded 4096-token, `d_model=64` eager baseline by **2.26× forward** and **2.28× backward**. The same baseline reached OOM at 16k tokens after requiring about 4.10 GiB before backward at 8k tokens. Full conditions and rows are in [the benchmark summary](assignment2-systems/benchmark_results/jit_attention/jit_attention_summary_eager_compiled_float32_noncausal.md) and [OOM table](assignment2-systems/benchmark_results/pytorch_attention/pytorch_attention_float32_noncausal.md).

### A5 — controlled GRPO ablations on Math12K

The primary alignment evidence is two controlled Math12K studies using the same model family, rule-based format/answer rewards, group size 8, learning rate `3e-5`, 200 GRPO steps, and batch/rollout batch sizes of 256. The first isolates group standard-deviation normalization: disabling it improves terminal accuracy from **48.4% to 61.4%**, with a corresponding increase in response length. The second isolates length normalization: the DAPO-style variant improves terminal accuracy from **39.0% to 48.2%** while reducing overall response length.

<p align="center">
  <img src="assignment5-alignment/docs/assets/grpo_std_accuracy.png" alt="GRPO advantage standard-deviation normalization ablation: evaluation accuracy" width="48%">
  <img src="assignment5-alignment/docs/assets/grpo_std_response_length.png" alt="GRPO advantage standard-deviation normalization ablation: average response length" width="48%">
</p>

<p align="center">
  <img src="assignment5-alignment/docs/assets/grpo_dapo_accuracy.png" alt="GRPO length-normalization ablation: evaluation accuracy" width="48%">
  <img src="assignment5-alignment/docs/assets/grpo_dapo_response_length.png" alt="GRPO length-normalization ablation: average response length" width="48%">
</p>

The [selected A5 results](assignment5-alignment/docs/results.md) record the exact final metrics, controlled variables, and interpretation limits. SFT and EI remain implemented and documented as additional alignment workflows, but use a separate GSM8K experiment track and are not compared directly with these Math12K GRPO results.

## Run locally

Install [uv](https://docs.astral.sh/uv/), then use an individual module environment:

```bash
# A1
cd assignment1-basics && uv run pytest

# A2
cd ../assignment2-systems && uv run pytest

# A5
cd ../assignment5-alignment && uv run pytest
```

A2's Triton benchmarks and A5's vLLM/FlashAttention paths require a CUDA-compatible environment. See each module's README for entry points and its design notes for data/model expectations.

The public release intentionally omits model-backed test weights. If a local
checkout needs the A1/A5 fixture-dependent tests, restore those small fixtures
from the private workspace before running the full test suites.

## Public-release policy

The repository excludes datasets, model weights, tokenized corpora, checkpoints, W&B exports, local logs, caches, credentials, course PDFs, and local development instructions. Small test fixtures and rendered figures remain where they are needed for correctness or evidence.

Before publishing, confirm that the included starter-code provenance permits your intended release and add a root license accordingly. Existing nested licenses attribute portions of the material to Stanford University.

## Resume summary

Built an end-to-end LLM engineering project covering from-scratch BPE/Transformer pretraining, attention and KV-cache systems optimization, and SFT/EI/GRPO alignment experiments for mathematical reasoning. The repository emphasizes testable implementations, measured system behavior, and documented experimental trade-offs rather than shipping opaque model artifacts.
