# CS336-LG

[中文](README.zh-CN.md) | [English](README.md)

### 从零开始的大语言模型训练、系统优化与对齐

这是一个面向工程实践的 LLM 项目，覆盖三层能力：从零实现 tokenizer 与 Transformer 语言模型；研究 attention、KV Cache 和训练系统的效率；使用 SFT、Expert Iteration 与 GRPO 进行数学推理对齐训练。

仓库按模块独立组织。Assignment 1、Assignment 2 和 Assignment 5 是主要展示内容，Assignment 3/4 作为补充分析和数据工程模块保留。

## 项目亮点

- **预训练基础：** byte-level BPE、Transformer 组件、语言模型训练、checkpoint 与自回归生成。
- **系统优化：** 分块 PyTorch/Triton attention、KV Cache decoding，以及教学型 DDP/FSDP/optimizer sharding。
- **对齐训练：** response-only SFT、规则数学奖励、Expert Iteration 与 GRPO 风格策略优化。
- **证据优先：** 保留测试、运行入口、精选曲线和 benchmark 表格；数据集、权重、checkpoint 和原始日志只保留在本地。

## 仓库结构

```text
.
├── assignment1-basics/       # BPE、Transformer LM、训练循环、测试与曲线
├── assignment2-systems/      # Attention、KV Cache、分布式系统与 benchmark
├── assignment3-scaling/      # 补充 scaling-law 分析
├── assignment4-data/         # 补充预训练数据过滤 pipeline
└── assignment5-alignment/    # SFT / EI / GRPO、评估与实验图表
```

每个模块拥有独立的 `pyproject.toml` 和 `uv.lock`，请在对应目录中安装和运行。

## 核心模块

| 模块 | 主要实现 | 文档 |
| --- | --- | --- |
| **A1 — 预训练基础** | BPE、tokenizer、RMSNorm/SwiGLU/RoPE、causal attention、Transformer LM、AdamW、训练与生成 | [模块说明](assignment1-basics/README.zh-CN.md) · [设计文档](assignment1-basics/docs/README.zh-CN.md) |
| **A2 — 系统优化** | 分块 attention、Triton forward、KV Cache、DDP、FSDP、optimizer sharding | [模块说明](assignment2-systems/README.zh-CN.md) · [设计文档](assignment2-systems/docs/README.zh-CN.md) |
| **A5 — 对齐训练** | SFT、EI、GRPO、规则奖励、rollout 与评估工具 | [模块说明](assignment5-alignment/README.zh-CN.md) · [设计文档](assignment5-alignment/docs/README.zh-CN.md) · [精选结果](assignment5-alignment/docs/results.zh-CN.md) |

## 精选证据

以下是已完成实验的记录，不是本次重新运行的结果。实验条件和限制写在对应模块中。

### A1 — Transformer 训练验证损失下降

完成的 30k-step 训练运行中，验证损失从约 6.8 下降到 5.x 区间。这里只展示训练健康度曲线；数据集、tokenized corpus 和 checkpoint 不进入公开仓库。

<p align="center"><img src="assignment1-basics/docs/assets/a1_validation_loss.png" alt="A1 Transformer 验证损失" width="900"></p>

### A2 — compiler baseline 与显存边界

在 batch size 8、float32、4096 token、`d_model=64` 的 non-causal attention 记录中，`torch.compile` 相对 eager forward 加速 **2.26×**，backward 加速 **2.28×**。普通 attention 在 16k token 时 OOM。完整条件见 [benchmark summary](assignment2-systems/benchmark_results/jit_attention/jit_attention_summary_eager_compiled_float32_noncausal.zh-CN.md) 和 [OOM 表](assignment2-systems/benchmark_results/pytorch_attention/pytorch_attention_float32_noncausal.zh-CN.md)。

### A5 — Math12K 上的受控 GRPO 消融

首页只展示两组严格对照：group standard-deviation normalization，以及 DAPO-style length normalization。前者的终点 accuracy 从 **48.4% 提升到 61.4%**，但输出变长；后者的终点 accuracy 从 **39.0% 提升到 48.2%**，同时降低整体输出长度。

<p align="center"><img src="assignment5-alignment/docs/assets/grpo_std_accuracy.png" alt="GRPO std normalization accuracy" width="48%"><img src="assignment5-alignment/docs/assets/grpo_std_response_length.png" alt="GRPO std normalization response length" width="48%"></p>
<p align="center"><img src="assignment5-alignment/docs/assets/grpo_dapo_accuracy.png" alt="GRPO length normalization accuracy" width="48%"><img src="assignment5-alignment/docs/assets/grpo_dapo_response_length.png" alt="GRPO length normalization response length" width="48%"></p>

详见 [A5 精选结果](assignment5-alignment/docs/results.zh-CN.md)。SFT/EI 仍然保留，但它们基于 GSM8K，不能直接当作 Math12K GRPO 的 baseline。

## 本地运行

安装 [uv](https://docs.astral.sh/uv/) 后，在模块目录中运行：

```bash
cd assignment1-basics && uv run pytest
cd ../assignment2-systems && uv run pytest
cd ../assignment5-alignment && uv run pytest
```

A2 Triton benchmark 与 A5 vLLM/FlashAttention 路径需要兼容的 CUDA 环境。

公开版本有意不包含模型权重型测试 fixture。如果本地 clone 需要运行 A1/A5 中依赖 fixture 的完整测试，请先从私有工作区恢复这些小型权重。

## 公开发布策略

仓库排除数据集、模型权重、tokenized corpus、checkpoint、W&B 导出、日志、缓存、凭证、课程 PDF 和本地开发说明。小型测试 fixture 与精选图表仅在有助于测试或展示证据时保留。

发布前请确认 starter code 的授权范围，并据此添加根许可证。现有部分嵌套许可证将代码版权归属于 Stanford University。

## 简历概述

构建覆盖从零 BPE/Transformer 预训练、attention 与 KV Cache 系统优化，以及数学推理 SFT/EI/GRPO 对齐的端到端 LLM 工程项目。项目强调可测试实现、可测量的系统行为和可解释的实验权衡，而不是提交黑盒模型产物。
