# Assignment 2 Experiment Runbook

This directory is the only home for repeatable assignment2 experiments. Scratch scripts should not live beside these entrypoints.

## Goals

The experiments should support three things:

- Validate correctness beyond unit tests with small smoke runs.
- Produce quantitative results for the final project report.
- Create resume-ready evidence for systems work: speed, memory, scaling, and tradeoffs.

## Directory Standard

Experiment outputs live under:

```text
experiments/results/<domain>/<semantic-run-id>/
```

Each run directory must contain:

- `metadata.json`: command config, hardware/software environment, and short git commit
- `run.md`: human-readable run sheet with config, environment, outputs, and notes
- `results.csv`: canonical tabular data for plotting and aggregation
- `results.json`: raw structured results
- `results.md`: quick human-readable preview

Run ids should be semantic first, for example:

```text
flash-attention_impl-torch+flash-triton_b4_seq128-1024_d32-128_float32_noncausal_cuda_<git>_<time>
distributed-memory_125m-params_world1-8_<git>_<time>
```

The timestamp is only a uniqueness suffix. The important distinguishing information should appear before it.

## Execution Order

Run experiments in this order. Do not skip the smoke stage; it catches broken output paths and bad configs cheaply.

### Stage 0. Correctness Gate

Purpose: verify the project is in a good state before collecting numbers.

Command:

```bash
uv run pytest
```

Success criterion:

- All tests pass.
- Record the pass result manually in the final report later; no experiment output directory is needed for this gate.

### Stage 1. Smoke Runs

Purpose: verify experiment scripts and result layout with tiny inputs.

Commands:

```bash
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cpu \
  --batch-size 1 \
  --seq-lengths 16 \
  --d-models 8 \
  --implementations torch flash_pytorch \
  --warmup-steps 1 \
  --timing-steps 1 \
  --run-label smoke
```

```bash
uv run python scripts/experiments/distributed_memory_accounting.py \
  --parameters-m 125 \
  --world-sizes 1 2 4 8 \
  --run-label smoke
```

Success criterion:

- Each run directory contains `metadata.json`, `run.md`, `results.csv`, `results.json`, and `results.md`.

### Stage 2. Attention Benchmark

Purpose: compare vanilla attention, `torch.compile`, PyTorch FlashAttention, and Triton FlashAttention.

Primary run, kernel comparison:

```bash
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cuda \
  --batch-size 4 \
  --seq-lengths 512 1024 2048 4096 \
  --d-models 64 128 \
  --implementations torch torch_compile flash_triton \
  --warmup-steps 5 \
  --timing-steps 20 \
  --run-label primary-large
```

Causal run, kernel comparison:

```bash
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cuda \
  --batch-size 4 \
  --seq-lengths 512 1024 2048 4096 \
  --d-models 64 128 \
  --implementations torch torch_compile flash_triton \
  --causal \
  --warmup-steps 5 \
  --timing-steps 20 \
  --run-label causal-large
```

Long-context stress run:

```bash
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cuda \
  --batch-size 1 \
  --seq-lengths 2048 4096 8192 16384 \
  --d-models 64 \
  --implementations torch torch_compile flash_triton \
  --warmup-steps 3 \
  --timing-steps 10 \
  --run-label long-context
```

PyTorch tiled reference run:

```bash
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cuda \
  --batch-size 2 \
  --seq-lengths 128 256 512 1024 \
  --d-models 64 \
  --implementations torch flash_pytorch flash_triton \
  --warmup-steps 3 \
  --timing-steps 8 \
  --run-label flash-pytorch-reference
```

Questions:

- Where does tiled FlashAttention become faster than vanilla attention?
- How does runtime scale with sequence length?
- Which implementation hits OOM first?
- How much does causal masking change runtime?
- Is the Triton forward advantage hidden by the current PyTorch/compiled backward?

Metrics to extract:

- Forward mean ms
- Backward mean ms
- Forward peak GiB
- Backward peak GiB
- Status: `ok`, `oom`, `error`, `skipped`

Expected interpretation:

- Vanilla attention should become expensive as sequence length grows because it materializes score/probability matrices.
- Triton FlashAttention should show better forward memory behavior, but this project implementation currently uses a compiled PyTorch backward that materializes large tensors, so backward may not show full FlashAttention benefits.

### Stage 3. Distributed Memory Accounting

Purpose: quantify persistent per-rank memory for DDP, sharded optimizer, and FSDP.

Commands:

```bash
uv run python scripts/experiments/distributed_memory_accounting.py \
  --parameters-m 125 \
  --world-sizes 1 2 4 8 \
  --run-label 125m
```

```bash
uv run python scripts/experiments/distributed_memory_accounting.py \
  --parameters-m 1000 \
  --world-sizes 1 2 4 8 \
  --run-label 1b
```

Questions:

- What memory remains fully replicated in DDP?
- What does sharding optimizer state save?
- What does FSDP save beyond sharded optimizer?
- Why do transient all-gather buffers and activations still matter?

Metrics to extract:

- Param GiB per rank
- Grad GiB per rank
- Optimizer state GiB per rank
- Total persistent GiB per rank

### Stage 4. Transformer Step Benchmark

Purpose: measure end-to-end training-step cost for a Transformer LM.

Required dimensions:

- Model size: small is the main benchmark; medium/large are stress candidates
- Context length: 512, 1024, 2048 for the main run
- Batch size: chosen to stress but not immediately OOM on the available GPU
- Precision: fp32, bf16 autocast
- Step type: forward only, forward + backward, full optimizer step

Primary command, realistic single-GPU run:

```bash
uv run python scripts/experiments/benchmark_transformer_step.py \
  --device cuda \
  --model-sizes small \
  --context-lengths 512 1024 2048 \
  --batch-size 2 \
  --precisions fp32 bf16 \
  --warmup-steps 3 \
  --timing-steps 8 \
  --run-label small-b2-long
```

OOM/stress boundary run:

```bash
uv run python scripts/experiments/benchmark_transformer_step.py \
  --device cuda \
  --model-sizes small medium \
  --context-lengths 1024 2048 \
  --batch-size 1 \
  --precisions fp32 bf16 \
  --warmup-steps 2 \
  --timing-steps 5 \
  --run-label stress-boundary
```

Smoke command:

```bash
uv run python scripts/experiments/benchmark_transformer_step.py \
  --device cpu \
  --model-sizes tiny \
  --context-lengths 16 \
  --batch-size 1 \
  --vocab-size 100 \
  --precisions fp32 \
  --warmup-steps 1 \
  --timing-steps 1 \
  --run-label smoke
```

Questions:

- How much time is spent in forward vs backward vs optimizer step?
- How much memory does bf16 save?
- Which context length first becomes impractical on the available GPU?

### Stage 5. Plotting and Aggregation

Purpose: convert selected `results.csv` files into report figures.

Command:

```bash
uv run python scripts/experiments/plot_results.py
```

Planned outputs:

- Attention forward runtime vs sequence length
- Attention backward runtime vs sequence length
- Attention peak memory vs sequence length
- Distributed persistent memory vs world size
- Transformer step runtime by phase

Rule:

- Plots must read from `results.csv`, not from Markdown tables.

## Current Entry Points

### Attention Implementations

Entry point:

```bash
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cuda \
  --batch-size 4 \
  --seq-lengths 512 1024 2048 4096 \
  --d-models 64 128 \
  --implementations torch torch_compile flash_triton \
  --warmup-steps 5 \
  --timing-steps 20
```

Questions:

- When does tiled FlashAttention beat vanilla attention?
- How does causal masking affect runtime?
- Where do vanilla attention or compiled attention hit memory limits?
- How does sequence length dominate runtime and memory?

### Distributed Memory Accounting

Entry point:

```bash
uv run python scripts/experiments/distributed_memory_accounting.py \
  --parameters-m 125 \
  --world-sizes 1 2 4 8
```

Questions:

- What memory does DDP replicate?
- What does sharded optimizer save?
- What additional memory does FSDP save?
- Which buffers are transient and therefore not captured by simple accounting?

### Transformer Step Benchmark

Entry point:

```bash
uv run python scripts/experiments/benchmark_transformer_step.py \
  --device cuda \
  --model-sizes small \
  --context-lengths 512 1024 2048 \
  --batch-size 2 \
  --precisions fp32 bf16 \
  --warmup-steps 3 \
  --timing-steps 8
```

Questions:

- How much time is spent in forward vs backward vs optimizer update?
- How does bf16 affect runtime and peak memory?
- Which model/context combinations are practical on the current GPU?

## Final Report Inputs

Do not write the final report until the benchmark results exist. The final report should be generated from:

- selected `results.csv` files
- plots produced from those CSVs
- notes on correctness tests and implementation tradeoffs

## Implementation TODO

- Add a small manifest or index file that lists selected runs for the final report.
