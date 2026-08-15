# 精选 GRPO 结果

[中文](results.zh-CN.md) | [English](results.md)

本页记录 README 展示的两组 Math12K GRPO 受控消融，是本地实验记录的匿名化摘要，不包含 W&B 链接、checkpoint、数据集或生成样本。

## 共享设置

两组使用同一 Qwen2.5-Math-1.5B 模型族、Math12K train/test split、规则 format/answer reward、group size 8、学习率 `3e-5`、seed 42、200 个 GRPO step、`train_batch_size=256`、`rollout_batch_size=256`、每个 rollout batch 一轮优化，最大 response 长度 1024。下表为每个条件单次完成运行的终点指标。

## 1. Advantage std normalization

唯一目标变量是 `use_std_normalization`；两组都使用 clipped GRPO 和 `mask_normalize`。

| 条件 | Accuracy | Format score | 平均长度 | 正确样本长度 | 错误样本长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 使用 group std | 48.4% | 84.6% | 730 | 504 | 943 |
| 不使用 group std | **61.4%** | **87.6%** | 963 | 759 | 1,287 |
| 差值（no-std − with-std） | **+13.0 pts** | **+3.0 pts** | +31.8% | +50.7% | +36.4% |

去除 group std 提高了终点 accuracy，但正确和错误回答都会变长。这是质量/长度 trade-off，不代表 no-std 在所有 GRPO 设置中都更好。

<p align="center"><img src="assets/grpo_std_accuracy.png" alt="GRPO std normalization accuracy" width="48%"><img src="assets/grpo_std_response_length.png" alt="GRPO std normalization response length" width="48%"></p>

## 2. DAPO-style length normalization

目标变量是 `length_norm_type`；两组都使用 clipped GRPO 和 group std normalization。

| 条件 | Accuracy | Format score | 平均长度 | 正确样本长度 | 错误样本长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mask_normalize` | 39.0% | 83.2% | 683 | 379 | 878 |
| `mask_dapo` | **48.2%** | **93.4%** | **623** | **495** | **743** |
| 差值（DAPO − mask） | **+9.2 pts** | **+10.2 pts** | **−8.8%** | +30.7% | **−15.5%** |

在该完成的消融中，DAPO-style normalization 让正确回答使用更多推理 token，同时降低错误回答的平均长度，并提高终点 accuracy。该结果只有单 seed、单训练预算，应理解为定向工程观察，而不是普适规律。

<p align="center"><img src="assets/grpo_dapo_accuracy.png" alt="GRPO length normalization accuracy" width="48%"><img src="assets/grpo_dapo_response_length.png" alt="GRPO length normalization response length" width="48%"></p>

## 范围说明

仓库还保留 GSM8K 上的 SFT/EI，以及 GRPO LR/loss 诊断。它们属于不同实验轨道或需要更多稳定性上下文，因此本页不声称 GSM8K SFT 是 Math12K GRPO 的直接 baseline。
