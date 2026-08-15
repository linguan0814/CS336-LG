# JIT Attention Benchmark (Non-Causal, Float32)

[中文](jit_attention_eager_compiled_float32_noncausal.zh-CN.md) | English

| impl | batch | seq_len | d_model | dtype | fwd ms | bwd ms | mem before bwd GiB | est. saved GiB | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager | 8 | 256 | 16 | float32 | 0.292 | 0.613 | 0.020 | 0.004 | ok |
| compiled | 8 | 256 | 16 | float32 | 0.154 | 0.606 | 0.020 | 0.004 | ok |
| eager | 8 | 1024 | 16 | float32 | 1.789 | 4.685 | 0.081 | 0.064 | ok |
| compiled | 8 | 1024 | 16 | float32 | 0.737 | 1.972 | 0.081 | 0.064 | ok |
| eager | 8 | 4096 | 16 | float32 | 28.122 | 72.267 | 1.026 | 1.008 | ok |
| compiled | 8 | 4096 | 16 | float32 | 11.241 | 31.909 | 1.026 | 1.008 | ok |
| eager | 8 | 256 | 64 | float32 | 0.250 | 0.550 | 0.022 | 0.006 | ok |
| compiled | 8 | 256 | 64 | float32 | 0.368 | 0.456 | 0.022 | 0.006 | ok |
| eager | 8 | 1024 | 64 | float32 | 1.781 | 4.554 | 0.088 | 0.070 | ok |
| compiled | 8 | 1024 | 64 | float32 | 0.843 | 2.070 | 0.088 | 0.070 | ok |
| eager | 8 | 4096 | 64 | float32 | 27.849 | 75.946 | 1.055 | 1.031 | ok |
| compiled | 8 | 4096 | 64 | float32 | 12.335 | 33.292 | 1.055 | 1.031 | ok |
