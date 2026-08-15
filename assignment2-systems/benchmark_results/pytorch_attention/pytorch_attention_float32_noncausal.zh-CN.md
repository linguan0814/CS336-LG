# PyTorch attention 显存与 OOM benchmark

[中文](pytorch_attention_float32_noncausal.zh-CN.md) | [English](pytorch_attention_float32_noncausal.md)

以下是 batch size 8、float32、non-causal PyTorch attention 的显存/延迟记录。普通 attention 在 16,384 token 的多组配置下 OOM；在 8,192 token、`d_model=64` 时，backward 前已使用约 4.095 GiB。

| batch | 序列长度 | d_model | 类型 | forward ms | backward ms | backward 前显存 GiB | 保存显存估计 GiB | 状态 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8 | 256 | 16 | float32 | 0.223 | 0.646 | 0.020 | 0.004 | ok |
| 8 | 1024 | 16 | float32 | 1.761 | 4.590 | 0.081 | 0.064 | ok |
| 8 | 4096 | 16 | float32 | 26.919 | 69.245 | 1.026 | 1.008 | ok |
| 8 | 8192 | 16 | float32 | 427.775 | 1596.847 | 4.036 | 4.016 | ok |
| 8 | 16384 | 16 | float32 | OOM | OOM | OOM | 16.031 | oom |
| 8 | 4096 | 64 | float32 | 28.316 | 74.130 | 1.055 | 1.031 | ok |
| 8 | 8192 | 64 | float32 | 471.116 | 1687.162 | 4.095 | 4.062 | ok |
| 8 | 16384 | 64 | float32 | OOM | OOM | OOM | 16.125 | oom |
