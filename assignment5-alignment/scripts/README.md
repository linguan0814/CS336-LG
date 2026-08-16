# Alignment experiment script guide

[中文](README.zh-CN.md) | English

The shell scripts in this directory are launch configurations for the completed A5 ablations. They are retained as reproducible references; rerunning the large jobs is optional.

| Script | Study |
| --- | --- |
| `prepare_gsm8k_sft.py` | Convert GSM8K-style records to the SFT format |
| `run_ei_ablation.sh` | Expert Iteration rollout-budget splits |
| `run_grpo_lr.sh` | GRPO learning-rate sensitivity |
| `run_grpo_offpolicy.sh` | Clipped GRPO comparison |
| `run_grpo_std.sh` | Group standard-deviation normalization |
| `run_grpo_length_norm.sh` | Length-normalization variants |
| `run_grpo_prompt.sh` | Prompt/reward-format variants |
| `run_grpo_all.sh` | Convenience wrapper for the GRPO studies |
| `evaluate_safety.py` | Evaluation utility requiring a local model path |

## Configuration conventions

The scripts pass explicit model, data, output, and CUDA-device arguments to the training modules. Set those paths for your own environment before running them. Outputs, checkpoints, logs, datasets, and model weights are ignored by Git.

Some script filenames reflect the original experiment sequence rather than a method guarantee. Read the arguments passed to `train_grpo.py`—especially `loss_type`, clipping-related settings, and rollout configuration—when interpreting an ablation. The retained plots and [A5 design notes](../docs/README.md) state the public evidence boundary.
