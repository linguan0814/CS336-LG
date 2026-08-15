# 系统优化

[中文](README.zh-CN.md) | [English](README.md)

本模块研究 LLM 训练和推理中的性能与显存行为，是仓库的系统方向核心展示：包括自定义 attention kernel、KV Cache decoding、分布式训练原语、测试和结构化 benchmark。

## 包含内容

- PyTorch 中基于 tile 和 online softmax 的 FlashAttention forward/backward。
- Triton FlashAttention forward kernel，支持 causal mask 与 log-sum-exp 状态；backward 当前使用编译后的 PyTorch 实现，并非完整 Triton backward kernel。
- KV Cache 分配与 cached decoding 工具，以及面向生成的 benchmark。
- 教学型 DDP、sharded optimizer 和 FSDP，实现同步与状态分片机制。
- 正确性测试，以及 `benchmark_results/` 中的 eager/`torch.compile` baseline。

## 设计概览

```text
attention baseline → tiled online-softmax → Triton forward kernel
         │
         ├→ 延迟/显存测量
         └→ KV Cache decoding 与分布式状态分片
```

模块会区分“已经测量的 baseline”和“已经实现但没有完整性能结论的 kernel”。具体边界见[设计文档](docs/README.zh-CN.md)。

```text
assignment2-systems/
├── cs336_systems/            # FlashAttention、KV Cache、DDP/FSDP、optimizer sharding
├── cs336-basics/             # A1 依赖实现
├── scripts/experiments/      # attention、KV Cache、Transformer-step benchmark
├── benchmark_results/        # 精选的轻量 benchmark 表格
├── tests/
├── pyproject.toml
└── uv.lock
```

## 运行

```bash
uv run pytest
uv run python scripts/experiments/benchmark_flash_attention.py --help
uv run python scripts/experiments/benchmark_kv_cache.py --help
```

后两个入口需要 CUDA/Triton 环境。生成的实验目录、缓存和日志保持本地；提交的 benchmark 表格用于复核已完成实验。
