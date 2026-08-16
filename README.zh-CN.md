# CS336-LG

[中文](README.zh-CN.md) | [English](README.md)

### 从零实现大语言模型训练、系统优化与对齐

CS336-LG 是一个模块化的大语言模型工程实现，覆盖 byte-level tokenizer 与 Transformer 预训练、attention 与 decoding 优化、scaling-law 分析、预训练数据过滤，以及使用 SFT、Expert Iteration 和 GRPO 进行 post-training。

仓库沿用 Stanford CS336 的主题划分，同时将各模块整理为可独立安装、测试和阅读的工程项目。使用者可以研究具体实现、复现已保留的测量结果，或在不下载完整训练产物的情况下扩展单个组件。

## 功能概览

- **从零预训练：** byte-level BPE、RMSNorm、RoPE、SwiGLU、causal multi-head attention、Transformer LM 训练、checkpoint 与自回归生成。
- **LLM 系统优化：** tiled attention、Triton attention kernel、KV Cache decoding、分布式训练原语，以及结构化延迟/显存 benchmark。
- **Scaling 与数据工程：** 基于公开 IsoFLOPs 曲线的拟合，以及包含语言、质量、PII 和去重阶段的网页文本过滤 pipeline。
- **对齐训练：** response-only SFT、Expert Iteration、规则数学奖励和 GRPO-style policy optimization。
- **可复现文档：** 模块级环境、测试、运行手册、benchmark 表格、受控消融和中英文文档。

## 仓库结构

```text
.
├── assignment1-basics/       # Tokenizer、Transformer LM、训练与生成
├── assignment2-systems/      # Attention、KV Cache、分布式系统与 benchmark
├── assignment3-scaling/      # 公开 IsoFLOPs scaling-law 分析
├── assignment4-data/         # 预训练数据提取与过滤 pipeline
└── assignment5-alignment/    # SFT、Expert Iteration、GRPO 与评估
```

每个模块都有独立的 `pyproject.toml` 和 `uv.lock`。系统优化与对齐模块的 CUDA/Python 依赖不同，因此环境有意保持隔离。

## 模块说明

| 模块 | 范围 | 文档 |
| --- | --- | --- |
| **A1 — 预训练基础** | BPE 训练、tokenizer、Transformer 组件、AdamW、训练循环、评估与生成 | [README](assignment1-basics/README.zh-CN.md) · [设计文档](assignment1-basics/docs/README.zh-CN.md) |
| **A2 — 系统优化** | Tiled/Triton attention、KV Cache decoding、DDP、FSDP、optimizer-state sharding 与 benchmark | [README](assignment2-systems/README.zh-CN.md) · [设计文档](assignment2-systems/docs/README.zh-CN.md) |
| **A3 — Scaling Laws** | 基于公开 IsoFLOPs 曲线的 compute-optimal 拟合 | [README](assignment3-scaling/README.zh-CN.md) · [报告](assignment3-scaling/docs/scaling_law_report.zh-CN.md) |
| **A4 — 数据工程** | HTML/WET 提取、语言过滤、PII masking、质量过滤与去重 | [README](assignment4-data/README.zh-CN.md) · [报告](assignment4-data/docs/data_filtering_report.zh-CN.md) |
| **A5 — 对齐训练** | SFT、Expert Iteration、GRPO、规则奖励、rollout 与评估 | [README](assignment5-alignment/README.zh-CN.md) · [设计文档](assignment5-alignment/docs/README.zh-CN.md) · [结果](assignment5-alignment/docs/results.zh-CN.md) |

## 快速开始

基础要求：

- CPU 组件可在 macOS 或 Linux 上运行；
- A1–A4 使用 Python 3.12；
- A5 使用 Python 3.11 或 3.12；
- 安装 [uv](https://docs.astral.sh/uv/)；
- Triton、FlashAttention、vLLM 和 GPU benchmark 需要支持 CUDA 的 Linux 环境。

克隆仓库后，每次进入一个模块安装环境：

```bash
git clone https://github.com/linguan0814/CS336-LG.git
cd CS336-LG/assignment1-basics

uv sync
uv run pytest
```

其余可在 CPU 环境运行的模块使用相同方式：

```bash
cd ../assignment2-systems && uv sync && uv run pytest
cd ../assignment3-scaling && uv sync
cd ../assignment4-data && uv sync && uv run pytest
```

A5 的锁定依赖包含 FlashAttention 与 vLLM，应在支持 CUDA 的 Linux 环境中安装：

```bash
cd ../assignment5-alignment && uv sync && uv run pytest
```

A1/A5 中有少量继承自课程环境的测试依赖本地模型权重 fixture。公开仓库有意排除这些权重；受影响的测试和依赖要求已在相应模块 README 中说明。

## 已记录结果

以下图表来自仓库中已有的完成运行，不是在整理仓库时重新生成。完整实验条件和解释限制见对应模块报告。

### A1 — Transformer 预训练

已记录的 30k-step 运行中，held-out validation loss 从约 6.8 下降到 5.x 中段。

<p align="center">
  <img src="assignment1-basics/docs/assets/a1_validation_loss.png" alt="A1 Transformer 验证损失" width="900">
</p>

### A2 — Attention baseline

在 batch size 8、sequence length 4096、`d_model=64` 的 non-causal float32 attention 测量中，`torch.compile` 相对 eager execution 获得 **2.26× forward** 和 **2.28× backward** 加速。PyTorch baseline 在 16k token 时 OOM；8k token 时 backward 前显存约为 4.10 GiB。

完整数据见 [compiled-attention 汇总](assignment2-systems/benchmark_results/jit_attention/jit_attention_summary_eager_compiled_float32_noncausal.zh-CN.md) 和 [显存/OOM 表](assignment2-systems/benchmark_results/pytorch_attention/pytorch_attention_float32_noncausal.zh-CN.md)。

### A5 — 受控 GRPO 消融

两组 Math12K 实验在匹配训练设置下分别研究 advantage standard-deviation normalization 与 DAPO-style length normalization：

- 关闭 group standard-deviation normalization：终点 accuracy **48.4% → 61.4%**，同时回答长度增加；
- 使用 DAPO-style length normalization：终点 accuracy **39.0% → 48.2%**，同时整体回答长度下降。

<p align="center">
  <img src="assignment5-alignment/docs/assets/grpo_std_accuracy.png" alt="GRPO std normalization accuracy" width="48%">
  <img src="assignment5-alignment/docs/assets/grpo_std_response_length.png" alt="GRPO std normalization response length" width="48%">
</p>

<p align="center">
  <img src="assignment5-alignment/docs/assets/grpo_dapo_accuracy.png" alt="GRPO length normalization accuracy" width="48%">
  <img src="assignment5-alignment/docs/assets/grpo_dapo_response_length.png" alt="GRPO length normalization response length" width="48%">
</p>

控制变量、正确/错误回答长度和解释限制见 [GRPO 精选结果](assignment5-alignment/docs/results.zh-CN.md)。SFT 与 Expert Iteration 使用独立的 GSM8K 实验轨道，不作为这些 Math12K 运行的直接 baseline。

## 可复现性与产物策略

仓库跟踪源码、配置、测试、轻量公开输入、精选图表和紧凑结果表。以下内容有意不进入公开版本：

- 原始数据集与 tokenized corpus；
- 模型权重与 checkpoint；
- W&B 运行目录与导出；
- 自动生成的实验输出、缓存和日志；
- 密钥、本地环境文件和机器相关配置。

需要凭证或外部产物的脚本通过本地路径或环境变量接收配置。根目录 `.gitignore` 定义完整的发布边界。

## 参与贡献

欢迎提交 bug fix、测试、文档和可复现性改进。修改应保持模块可独立运行，不提交生成产物；新增性能或实验结论时应附带可核验的测量依据。

## 许可证与归属

本仓库基于 Stanford CS336 的课程结构和部分 starter material。各模块中的嵌套许可证适用于相应内容。在重新分发或为整个仓库添加统一许可证前，请先核对上游课程条款和现有模块许可证。
