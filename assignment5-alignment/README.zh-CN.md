# 对齐训练

[中文](README.zh-CN.md) | [English](README.md)

这是面向数学推理的 post-training 模块，包含 supervised fine-tuning（SFT）、Expert Iteration（EI）和基于规则奖励的 GRPO-style policy optimization。

## 包含内容

- prompt 构造、response-only SFT loss、数据准备、rollout 与评估工具。
- 检查 `<think>...</think><answer>...</answer>` 格式，以及数学答案等价性的规则奖励。
- `cs336_alignment/` 中的 SFT、EI、GRPO 训练入口。
- 小型模型/tokenizer fixture 与单元测试。
- `docs/assets/` 中的精选实验图，以及 `docs/results.md` 中的匿名化 GRPO 结果表。

一个 DPO fixture 测试使用的 tiny GPT-2 权重仅保留在本地，不进入公开仓库；tokenizer fixture 和测试源码仍然保留供复盘。

## 设计概览

```text
prompt + verifier → rollout → format / answer rewards
                         ├→ SFT：只在 response token 上训练
                         ├→ EI：筛选验证通过的轨迹再 SFT
                         └→ GRPO：使用 group-relative reward 更新策略
```

SFT/EI 使用 GSM8K 实验轨道；首页 GRPO 证据使用 Math12K 轨道，二者不做直接 baseline 比较。完整控制变量和结果见[设计文档](docs/README.zh-CN.md)与[精选结果](docs/results.zh-CN.md)。

```text
assignment5-alignment/
├── cs336_alignment/          # 训练循环、奖励、数据、prompt 与评估
├── scripts/                  # 数据准备、评估和 reporting 工具
├── tests/                    # 测试与小型 fixture
├── docs/                     # 设计文档、精选结果与证据图
├── pyproject.toml
└── uv.lock
```

## 运行

```bash
uv run pytest
uv run python -m cs336_alignment.train_sft_step --help
uv run python -m cs336_alignment.train_ei_step --help
uv run python -m cs336_alignment.train_grpo --help
```

完整训练需要本地数据、模型权重和兼容的 CUDA/vLLM/FlashAttention 环境。checkpoint、原始 W&B 导出和运行日志不会进入 Git。
