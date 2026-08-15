# A5 design notes

[中文](README.zh-CN.md) | English

## Objective

Study post-training methods for verifiable mathematical reasoning, using a common prompt/reward contract across supervised and reinforcement-style updates.

## Training paths

| Path | Main code | Training signal |
| --- | --- | --- |
| SFT | `train_sft_step.py`, `sft_utils.py` | Next-token loss applied to response tokens |
| Expert Iteration | `train_ei_step.py` | Generate candidates, verify them, then fine-tune on accepted trajectories |
| GRPO | `train_grpo.py`, `grpo_utils.py` | Group-relative rewards from rollout samples |
| Evaluation | `drgrpo_grader.py`, `metrics.py` | Structured-format checks and mathematical-answer verification |

## Experiment interpretation

The repository contains two separate experiment tracks and they must not be conflated:

- **SFT and EI:** GSM8K experiments that demonstrate the supervised and iterative-training workflows.
- **GRPO:** Math12K experiments that study reward-driven policy updates and their optimization choices.

The public headline evidence is deliberately limited to two matched GRPO ablations: group standard-deviation normalization and DAPO-style length normalization. Their exact final metrics, shared settings, and caveats are recorded in [the selected result table](results.md). Learning-rate and loss-function plots are retained locally as diagnostic artifacts, but are not presented as headline conclusions.

All figures are historical experiment artifacts, not leaderboard claims or a replacement for rerunning the jobs. The raw W&B exports, model weights, checkpoints, and datasets are intentionally retained only locally.

## Reproduction boundary

Use `uv run pytest` for the committed tests. Full SFT/EI/GRPO runs require locally supplied data and model weights plus a compatible CUDA/vLLM/FlashAttention environment. See [`scripts/README.md`](../scripts/README.md) for the script map and configuration conventions.
