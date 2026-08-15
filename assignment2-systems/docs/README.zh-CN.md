# A2 设计文档

[中文](README.zh-CN.md) | [English](README.md)

## 目标

让 LLM 训练和 decoding 的系统权衡可检查：attention 的二次显存、kernel/compiler 开销、cached generation，以及分布式 optimizer state。

## 实现映射

| 组件 | 代码 | 关注点 |
| --- | --- | --- |
| 分块 attention | `cs336_systems/flash_attention/flash_att_pytorch.py` | Q/K tile 上的 online softmax，减少完整 score matrix 物化 |
| Triton attention | `cs336_systems/flash_attention/flash_att_triton.py` | causal mask 与 log-sum-exp 状态的 forward kernel |
| KV Cache | `cs336_systems/kv_cache/cache.py` | 自回归 decoding 中复用 key/value state |
| 分布式状态 | `ddp.py`、`fsdp.py`、`sharded_optimizer.py` | 同步与 sharding 的教学实现 |
| Benchmark | `scripts/experiments/` | attention 和 decoding 的延迟/显存测量 |

## 已提交测量的含义

benchmark 表格建立了 eager 与 `torch.compile` baseline，并记录了普通 attention 的长上下文 OOM 边界。它们**没有**宣称完成端到端 Triton-forward 对比；公开结论严格匹配已有实验覆盖范围。

- [Eager vs. compiled summary](../benchmark_results/jit_attention/jit_attention_summary_eager_compiled_float32_noncausal.zh-CN.md)
- [PyTorch attention 显存/OOM 表](../benchmark_results/pytorch_attention/pytorch_attention_float32_noncausal.zh-CN.md)

## 复现边界

`uv run pytest` 可运行单元测试。Triton 路径和性能实验需要 CUDA；具体耗时会受到 GPU、驱动、PyTorch/Triton 版本、时钟和显存影响。
