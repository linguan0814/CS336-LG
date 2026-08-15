# JIT Attention Benchmark Summary

[中文](jit_attention_summary_eager_compiled_float32_noncausal.zh-CN.md) | English

| seq_len | d_model | dtype | eager fwd ms | compiled fwd ms | fwd speedup | eager bwd ms | compiled bwd ms | bwd speedup | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 16 | float32 | 0.292 | 0.154 | 1.897 | 0.613 | 0.606 | 1.012 | compiled:ok,eager:ok |
| 256 | 64 | float32 | 0.250 | 0.368 | 0.680 | 0.550 | 0.456 | 1.205 | compiled:ok,eager:ok |
| 1024 | 16 | float32 | 1.789 | 0.737 | 2.425 | 4.685 | 1.972 | 2.376 | compiled:ok,eager:ok |
| 1024 | 64 | float32 | 1.781 | 0.843 | 2.114 | 4.554 | 2.070 | 2.200 | compiled:ok,eager:ok |
| 4096 | 16 | float32 | 28.122 | 11.241 | 2.502 | 72.267 | 31.909 | 2.265 | compiled:ok,eager:ok |
| 4096 | 64 | float32 | 27.849 | 12.335 | 2.258 | 75.946 | 33.292 | 2.281 | compiled:ok,eager:ok |
