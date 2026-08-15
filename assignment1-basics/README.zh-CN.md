# 预训练基础

[中文](README.zh-CN.md) | [English](README.md)

这是仓库的基础能力模块：不依赖高级模型框架，从原始文本到训练和生成，搭建一个小型 causal Transformer language model。

## 包含内容

- GPT 风格正则预分词、byte-level BPE 训练、序列化、encode/decode。
- Embedding、Linear、RMSNorm、SwiGLU、RoPE、scaled dot-product attention、多头 attention 与 pre-norm Transformer。
- memmap 数据加载、AdamW、学习率调度、梯度裁剪、验证、checkpoint 和采样生成。
- 覆盖 tokenizer、BPE、序列化、数据、优化器、模型组件和 Transformer LM 的测试。

依赖模型权重的测试 fixture 仅保留在本地，根目录 `.gitignore` 会将其排除；tokenizer 和组件级测试不依赖该权重。

## 设计概览

```text
原始文本 → BPE 训练 → tokenizer 产物 → tokenized memmap
                                      ↓
                           Transformer LM → 训练/验证 → 生成
```

`model/` 负责神经网络组件，`trainer/` 负责优化和数据工具，顶层模块提供 tokenizer、训练和生成入口。具体文件映射见[设计文档](docs/README.zh-CN.md)。

```text
assignment1-basics/
├── cs336_basics/             # tokenizer、模型、训练与生成
├── tests/                    # 单元测试与小型 fixture
├── docs/                     # 设计文档与精选训练曲线
├── scripts/                  # 报告辅助脚本
├── pyproject.toml
└── uv.lock
```

## 运行

```bash
uv run pytest
```

数据准备、tokenizer 训练、模型训练和生成入口位于 `cs336_basics/`。数据集、tokenized corpus、checkpoint 与 W&B 原始记录不提交。精选验证损失曲线位于 `docs/assets/`。

![A1 验证损失](docs/assets/a1_validation_loss.png)
