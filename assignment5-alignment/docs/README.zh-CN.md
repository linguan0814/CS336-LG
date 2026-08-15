# A5 设计文档

[中文](README.zh-CN.md) | [English](README.md)

## 目标

围绕可验证的数学推理研究 post-training 方法，并让 SFT、EI、GRPO 共享统一的 prompt/reward contract。

## 训练路径

| 路径 | 主要代码 | 训练信号 |
| --- | --- | --- |
| SFT | `train_sft_step.py`、`sft_utils.py` | 只在 response token 上计算 next-token loss |
| Expert Iteration | `train_ei_step.py` | 采样候选、验证答案，再对通过的轨迹 fine-tune |
| GRPO | `train_grpo.py`、`grpo_utils.py` | 使用 rollout 样本的 group-relative reward |
| 评估 | `drgrpo_grader.py`、`metrics.py` | 格式检查与数学答案验证 |

## 实验解读

SFT/EI 是 GSM8K 实验；GRPO 是 Math12K 实验。公开首页只展示两组匹配的 GRPO 消融：group std normalization 与 DAPO-style length normalization。精确终点指标、共享设置和 caveat 见[精选结果](results.zh-CN.md)。LR 和 loss 曲线作为本地诊断档案保留，不作为首页结论。

所有图表都是历史实验产物，不是 leaderboard 声明，也不能替代重新运行。原始 W&B、权重、checkpoint 和数据集只保留在本地。
