# JIT attention benchmark（non-causal，float32）

[中文](jit_attention_eager_compiled_float32_noncausal.zh-CN.md) | [English](jit_attention_eager_compiled_float32_noncausal.md)

以下是 batch size 8、float32、non-causal attention 下 eager 与 `torch.compile` 的原始明细。时间单位为毫秒，显存单位为 GiB。

| 实现 | batch | 序列长度 | d_model | 类型 | forward ms | backward ms | backward 前显存 | 保存显存估计 | 状态 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| eager | 8 | 256 | 16 | float32 | 0.292 | 0.613 | 0.020 | 0.004 | ok |
| compiled | 8 | 256 | 16 | float32 | 0.154 | 0.606 | 0.020 | 0.004 | ok |
| eager | 8 | 1024 | 16 | float32 | 1.789 | 4.685 | 0.081 | 0.064 | ok |
| compiled | 8 | 1024 | 16 | float32 | 0.737 | 1.972 | 0.081 | 0.064 | ok |
| eager | 8 | 4096 | 16 | float32 | 28.122 | 72.267 | 1.026 | 1.008 | ok |
| compiled | 8 | 4096 | 16 | float32 | 11.241 | 31.909 | 1.026 | 1.008 | ok |
| eager | 8 | 4096 | 64 | float32 | 27.849 | 75.946 | 1.055 | 1.031 | ok |
| compiled | 8 | 4096 | 64 | float32 | 12.335 | 33.292 | 1.055 | 1.031 | ok |
