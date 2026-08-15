# JIT attention benchmark 汇总

[中文](jit_attention_summary_eager_compiled_float32_noncausal.zh-CN.md) | [English](jit_attention_summary_eager_compiled_float32_noncausal.md)

| 序列长度 | d_model | 类型 | eager forward ms | compiled forward ms | forward 加速 | eager backward ms | compiled backward ms | backward 加速 | 状态 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 16 | float32 | 0.292 | 0.154 | 1.897x | 0.613 | 0.606 | 1.012x | ok |
| 256 | 64 | float32 | 0.250 | 0.368 | 0.680x | 0.550 | 0.456 | 1.205x | ok |
| 1024 | 16 | float32 | 1.789 | 0.737 | 2.425x | 4.685 | 1.972 | 2.376x | ok |
| 1024 | 64 | float32 | 1.781 | 0.843 | 2.114x | 4.554 | 2.070 | 2.200x | ok |
| 4096 | 16 | float32 | 28.122 | 11.241 | 2.502x | 72.267 | 31.909 | 2.265x | ok |
| 4096 | 64 | float32 | 27.849 | 12.335 | 2.258x | 75.946 | 33.292 | 2.281x | ok |
