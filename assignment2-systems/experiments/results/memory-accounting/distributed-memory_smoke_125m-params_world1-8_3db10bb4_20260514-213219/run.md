# Distributed Memory Accounting Run

## Run

- Run ID: `distributed-memory_smoke_125m-params_world1-8_3db10bb4_20260514-213219`
- Git commit: `3db10bb4`

## Configuration

| Key | Value |
| --- | --- |
| `output_dir` | `experiments/results` |
| `parameters_m` | `125.0000` |
| `run_id` | `distributed-memory_smoke_125m-params_world1-8_3db10bb4_20260514-213219` |
| `run_label` | `smoke` |
| `world_sizes` | `[1, 2, 4, 8]` |

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

- This is theoretical accounting for persistent parameter, gradient, and AdamW state memory.
- Transient communication buffers and activation memory are not included.
