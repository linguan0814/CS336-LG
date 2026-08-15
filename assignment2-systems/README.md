# Systems Optimization

[中文](README.zh-CN.md) | English

This module investigates the performance and memory behavior of LLM training and decoding. It is the main systems showcase in the repository, combining custom attention kernels, KV-cache decoding, distributed-training primitives, tests, and structured benchmarks.

## Included components

- Tiled online-softmax FlashAttention forward/backward implementation in PyTorch.
- Triton FlashAttention forward kernel, including causal masking and log-sum-exp state. Its backward path currently uses compiled PyTorch rather than a full Triton backward kernel.
- KV-cache allocation and cached decoding utilities, with benchmark scripts for generation-oriented comparisons.
- Instructional DDP, sharded-optimizer, and FSDP implementations that expose communication and state-sharding mechanics.
- Correctness tests plus eager/`torch.compile` attention baseline results in `benchmark_results/`.

## Design at a glance

```text
attention baseline ──→ tiled online-softmax attention ──→ Triton forward kernel
       │
       ├──→ benchmark methodology and recorded memory / latency measurements
       └──→ KV-cache decoding and distributed-state-sharding exercises
```

The module distinguishes measured baselines from implementation goals. In particular, the Triton forward path is implemented, while backward delegates to compiled PyTorch. [Design notes](docs/README.md) describe the boundary and link each benchmark to the appropriate implementation.

```text
assignment2-systems/
├── cs336_systems/            # FlashAttention, KV cache, DDP/FSDP, optimizer sharding
├── cs336-basics/             # Local A1 dependency used by the systems module
├── scripts/experiments/      # Attention, KV-cache, and Transformer-step benchmarks
├── benchmark_results/        # Curated lightweight benchmark tables
├── tests/
├── pyproject.toml
└── uv.lock
```

## Run

```bash
uv run pytest

# Requires a CUDA/Triton-compatible environment.
uv run python scripts/experiments/benchmark_flash_attention.py --help
uv run python scripts/experiments/benchmark_kv_cache.py --help
```

Benchmark scripts produce local structured run directories. Only the small, reviewable baseline tables under `benchmark_results/` are committed; generated experiment outputs, caches, and logs remain local.
