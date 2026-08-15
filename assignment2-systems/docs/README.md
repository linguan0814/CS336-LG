# A2 design notes

[中文](README.zh-CN.md) | English

## Objective

Make the systems trade-offs behind LLM training and decoding inspectable: quadratic attention memory, kernel/compile overhead, cached generation, and distributed optimizer state.

## Implementation map

| Component | Code | Design focus |
| --- | --- | --- |
| Tiled attention | `cs336_systems/flash_attention/flash_att_pytorch.py` | Online softmax over Q/K tiles without materializing the full score matrix in the tiled forward path |
| Triton attention | `cs336_systems/flash_attention/flash_att_triton.py` | Triton forward kernel with causal masking and log-sum-exp state |
| KV cache | `cs336_systems/kv_cache/cache.py` | Reuse key/value states during autoregressive decoding |
| Distributed state | `ddp.py`, `fsdp.py`, `sharded_optimizer.py` | Teaching implementations of synchronization and sharding mechanics |
| Benchmarking | `scripts/experiments/` | Structured latency and memory measurements for attention and decoding |

## What the committed measurements show

The committed benchmark tables establish an eager and `torch.compile` baseline, including a long-context OOM boundary for standard attention. They do **not** present a completed end-to-end Triton-forward versus baseline performance claim. This distinction is intentional: the public evidence should match the recorded experiment coverage.

- [Eager vs. compiled summary](../benchmark_results/jit_attention/jit_attention_summary_eager_compiled_float32_noncausal.md)
- [PyTorch attention memory/OOM table](../benchmark_results/pytorch_attention/pytorch_attention_float32_noncausal.md)

## Reproduction boundary

The unit tests can be run with `uv run pytest`. CUDA-compatible hardware is required for the Triton path and performance experiments; timings vary with GPU, driver, PyTorch/Triton version, clocks, and memory availability. Generated benchmark run directories are kept local, while the compact review tables above are versioned.
