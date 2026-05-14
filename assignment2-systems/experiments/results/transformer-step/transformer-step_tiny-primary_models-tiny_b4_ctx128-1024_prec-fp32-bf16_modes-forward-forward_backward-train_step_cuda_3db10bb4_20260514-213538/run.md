# Transformer Step Benchmark Run

## Run

- Run ID: `transformer-step_tiny-primary_models-tiny_b4_ctx128-1024_prec-fp32-bf16_modes-forward-forward_backward-train_step_cuda_3db10bb4_20260514-213538`
- Git commit: `3db10bb4`

## Configuration

| Key | Value |
| --- | --- |
| `batch_size` | `4` |
| `context_lengths` | `[128, 256, 512, 1024]` |
| `device` | `cuda` |
| `learning_rate` | `0.0010` |
| `model_sizes` | `['tiny']` |
| `modes` | `['forward', 'forward_backward', 'train_step']` |
| `output_dir` | `experiments/results` |
| `precisions` | `['fp32', 'bf16']` |
| `rope_theta` | `10000.0000` |
| `run_id` | `transformer-step_tiny-primary_models-tiny_b4_ctx128-1024_prec-fp32-bf16_modes-forward-forward_backward-train_step_cuda_3db10bb4_20260514-213538` |
| `run_label` | `tiny-primary` |
| `timing_steps` | `8` |
| `vocab_size` | `10000` |
| `warmup_steps` | `3` |

## Environment

| Key | Value |
| --- | --- |
| `cuda_available` | `True` |
| `cuda_runtime` | `13.0` |
| `device` | `cuda` |
| `git_short_commit` | `3db10bb4` |
| `gpu_name` | `NVIDIA GeForce RTX 4060 Laptop GPU` |
| `gpu_total_memory_gib` | `7.9956` |
| `torch` | `2.11.0+cu130` |

## Outputs

- `metadata.json`
- `run.md`
- `results.csv`
- `results.json`
- `results.md`

## Notes

- Forward mode includes logits and cross-entropy loss construction.
- Forward-backward mode excludes optimizer.step().
- Train-step mode includes AdamW step; warmup allocates AdamW state before timed iterations.
