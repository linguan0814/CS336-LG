# 对齐实验脚本说明

[中文](README.zh-CN.md) | [English](README.md)

这里的 shell 脚本是已完成 A5 消融实验的启动配置，用于复盘和复现入口，不要求 reviewer 重新运行大规模任务。

| 脚本 | 实验 |
| --- | --- |
| `prepare_gsm8k_sft.py` | 将 GSM8K 样本转成 SFT 格式 |
| `run_ei_ablation.sh` | Expert Iteration rollout budget 划分 |
| `run_grpo_lr.sh` | GRPO 学习率敏感性 |
| `run_grpo_offpolicy.sh` | clipped GRPO 对照 |
| `run_grpo_std.sh` | group std normalization |
| `run_grpo_length_norm.sh` | length normalization 变体 |
| `run_grpo_prompt.sh` | prompt/reward 格式变体 |
| `run_grpo_all.sh` | GRPO 实验组合入口 |
| `evaluate_safety.py` | 需要本地模型路径的评估工具 |

运行前请根据自己的环境设置模型、数据、输出目录和 CUDA device。输出、checkpoint、日志、数据集和模型权重均被 Git 忽略。部分脚本文件名沿用原实验顺序，解释消融时应以传给 `train_grpo.py` 的 `loss_type`、clipping、rollout 参数为准。
