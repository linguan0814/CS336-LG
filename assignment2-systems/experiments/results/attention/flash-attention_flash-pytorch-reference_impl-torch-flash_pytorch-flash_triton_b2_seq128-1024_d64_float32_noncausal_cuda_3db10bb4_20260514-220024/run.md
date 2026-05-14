# Flash Attention Benchmark Run

## Run

- Run ID: `flash-attention_flash-pytorch-reference_impl-torch-flash_pytorch-flash_triton_b2_seq128-1024_d64_float32_noncausal_cuda_3db10bb4_20260514-220024`
- Git commit: `3db10bb4`

## Configuration

| Key | Value |
| --- | --- |
| `batch_size` | `2` |
| `causal` | `False` |
| `d_models` | `[64]` |
| `device` | `cuda` |
| `dtype` | `float32` |
| `implementations` | `['torch', 'flash_pytorch', 'flash_triton']` |
| `output_dir` | `experiments/results` |
| `run_id` | `flash-attention_flash-pytorch-reference_impl-torch-flash_pytorch-flash_triton_b2_seq128-1024_d64_float32_noncausal_cuda_3db10bb4_20260514-220024` |
| `run_label` | `flash-pytorch-reference` |
| `seq_lengths` | `[128, 256, 512, 1024]` |
| `timing_steps` | `8` |
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

- CSV and JSON are canonical machine-readable records.
- Markdown results are a quick preview and should not be used as the source for plots.
