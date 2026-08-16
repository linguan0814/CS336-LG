# A2 实验计划

[中文](EXPERIMENT_PLAN.zh-CN.md) | [English](EXPERIMENT_PLAN.md)

这份 runbook 规定 A2 attention、KV Cache 和 Transformer-step benchmark 的执行顺序、正确性门禁和输出格式。它是实验参考，不要求使用者重新运行大规模 CUDA 任务。

## 1. 实验目标

- 在 benchmark 前验证 PyTorch attention、Triton forward 和 KV Cache 的正确性。
- 测量 eager、`torch.compile`、普通 attention 与 FlashAttention 路径的延迟和显存行为。
- 记录 KV Cache 在 prompt + generation 场景下的性能变化。
- 估算 DDP、sharded optimizer 与 FSDP 的 per-rank persistent memory。

## 2. 正确性门禁

先在 `assignment2-systems/` 运行：

```bash
uv run pytest
```

只有测试通过后才运行 CUDA benchmark。结果必须包含实现名称、设备、dtype、batch、序列长度、warmup/timing steps 和状态字段；OOM 也应作为结构化状态记录，而不是被静默丢弃。

## 3. Benchmark 类型

| 类型 | 入口 | 主要问题 |
| --- | --- | --- |
| Attention | `benchmark_flash_attention.py` | 不同实现的延迟、显存与 causal/non-causal 行为 |
| KV Cache | `benchmark_kv_cache.py` | cached decoding 相对 no-cache 的 token latency 和 memory |
| Transformer step | `benchmark_transformer_step.py` | 模型规模、上下文长度和 dtype 对训练 step 的影响 |
| Distributed memory | `distributed_memory_accounting.py` | DDP、sharding 和 FSDP 的理论 persistent memory |

## 4. 输出约定

每次运行可生成：

```text
metadata.json
run.md
results.csv
results.json
results.md
```

这些运行目录属于本地实验档案。公开仓库只保留小型、可审阅的 baseline 表格；模型、数据、日志和完整运行目录通过 `.gitignore` 排除。

## 5. 解读边界

已有公开 benchmark 主要建立 eager/`torch.compile` baseline 和普通 attention 的长上下文 OOM 边界。Triton forward 已实现，但当前 backward 仍使用 compiled PyTorch；因此 README 不把现有文件包装成完整 Triton forward/backward 性能结论。

GPU、驱动、PyTorch/Triton 版本、时钟频率和显存状态都会影响绝对耗时。报告应优先比较同一运行条件下的相对趋势，并保留异常和 OOM 状态。
