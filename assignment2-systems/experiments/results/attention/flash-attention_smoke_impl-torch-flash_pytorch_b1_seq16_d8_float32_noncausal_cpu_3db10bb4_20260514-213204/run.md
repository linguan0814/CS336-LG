# Flash Attention Benchmark Run

## Run

- Run ID: `flash-attention_smoke_impl-torch-flash_pytorch_b1_seq16_d8_float32_noncausal_cpu_3db10bb4_20260514-213204`
- Git commit: `3db10bb4`

## Configuration

| Key | Value |
| --- | --- |
| `batch_size` | `1` |
| `causal` | `False` |
| `d_models` | `[8]` |
| `device` | `cpu` |
| `dtype` | `float32` |
| `implementations` | `['torch', 'flash_pytorch']` |
| `output_dir` | `experiments/results` |
| `run_id` | `flash-attention_smoke_impl-torch-flash_pytorch_b1_seq16_d8_float32_noncausal_cpu_3db10bb4_20260514-213204` |
| `run_label` | `smoke` |
| `seq_lengths` | `[16]` |
| `timing_steps` | `1` |
| `warmup_steps` | `1` |

## Environment

| Key | Value |
| --- | --- |
| `cuda_available` | `True` |
| `device` | `cpu` |
| `git_short_commit` | `3db10bb4` |
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
