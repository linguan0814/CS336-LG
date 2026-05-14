# Assignment 2 Experiment Plan

This folder contains repeatable experiment entrypoints. Keep exploratory scratch scripts in `scripts/` until they are promoted here.

## Runs to Collect

1. Attention implementation comparison
   - `torch`
   - `torch_compile`
   - `flash_pytorch`
   - `flash_triton`
   - vary sequence length, hidden dimension, dtype, and causal masking

2. Transformer training step benchmark
   - forward only
   - forward + backward
   - full optimizer step
   - fp32 vs bf16 autocast
   - activation checkpointing on/off

3. Distributed systems analysis
   - DDP correctness and communication pattern
   - sharded optimizer memory accounting
   - FSDP parameter/gradient sharding accounting

## Output Convention

Each promoted experiment should write:

- `metadata.json`: config and hardware/software environment
- `results.csv`: tabular data for plotting
- `results.json`: structured raw results
- `results.md`: quick human-readable table
