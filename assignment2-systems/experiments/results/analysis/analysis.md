# Experiment Analysis Index

## Included Data

- attention: 186 rows from 9 runs
- memory-accounting: 36 rows from 3 runs
- transformer-step: 66 rows from 5 runs

## Figures

- attention_backward_d64_causal: [figures/attention_backward_d64_causal.png](figures/attention_backward_d64_causal.png)
- attention_backward_d64_noncausal: [figures/attention_backward_d64_noncausal.png](figures/attention_backward_d64_noncausal.png)
- attention_flash_pytorch_small_forward_log: [figures/attention_flash_pytorch_small_forward_log.png](figures/attention_flash_pytorch_small_forward_log.png)
- attention_forward_d64_causal: [figures/attention_forward_d64_causal.png](figures/attention_forward_d64_causal.png)
- attention_forward_d64_noncausal: [figures/attention_forward_d64_noncausal.png](figures/attention_forward_d64_noncausal.png)
- attention_forward_speedup_d64_causal: [figures/attention_forward_speedup_d64_causal.png](figures/attention_forward_speedup_d64_causal.png)
- attention_forward_speedup_d64_noncausal: [figures/attention_forward_speedup_d64_noncausal.png](figures/attention_forward_speedup_d64_noncausal.png)
- distributed_memory_1000m: [figures/distributed_memory_1000m.png](figures/distributed_memory_1000m.png)
- distributed_memory_125m: [figures/distributed_memory_125m.png](figures/distributed_memory_125m.png)
- transformer_step_medium_bf16: [figures/transformer_step_medium_bf16.png](figures/transformer_step_medium_bf16.png)
- transformer_step_medium_fp32: [figures/transformer_step_medium_fp32.png](figures/transformer_step_medium_fp32.png)
- transformer_step_memory_medium_bf16: [figures/transformer_step_memory_medium_bf16.png](figures/transformer_step_memory_medium_bf16.png)
- transformer_step_memory_medium_fp32: [figures/transformer_step_memory_medium_fp32.png](figures/transformer_step_memory_medium_fp32.png)
- transformer_step_memory_small_bf16: [figures/transformer_step_memory_small_bf16.png](figures/transformer_step_memory_small_bf16.png)
- transformer_step_memory_small_fp32: [figures/transformer_step_memory_small_fp32.png](figures/transformer_step_memory_small_fp32.png)
- transformer_step_memory_tiny_bf16: [figures/transformer_step_memory_tiny_bf16.png](figures/transformer_step_memory_tiny_bf16.png)
- transformer_step_memory_tiny_fp32: [figures/transformer_step_memory_tiny_fp32.png](figures/transformer_step_memory_tiny_fp32.png)
- transformer_step_small_bf16: [figures/transformer_step_small_bf16.png](figures/transformer_step_small_bf16.png)
- transformer_step_small_fp32: [figures/transformer_step_small_fp32.png](figures/transformer_step_small_fp32.png)
- transformer_step_tiny_bf16: [figures/transformer_step_tiny_bf16.png](figures/transformer_step_tiny_bf16.png)
- transformer_step_tiny_fp32: [figures/transformer_step_tiny_fp32.png](figures/transformer_step_tiny_fp32.png)

## Notes

- Figures are generated from `results.csv` files only.
- Attention backward numbers for `flash_triton` reflect the current project implementation, whose backward path uses compiled PyTorch.
- Distributed memory accounting is theoretical persistent memory and excludes activations and transient communication buffers.
