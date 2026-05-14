| implementation | batch_size | seq_len | d_model | dtype | causal | forward_mean_ms | forward_std_ms | backward_mean_ms | backward_std_ms | forward_peak_gib | backward_peak_gib | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | 2 | 128 | 32 | float32 | False | 0.3278 | 0.1091 | 0.6755 | 0.1171 | 0.0083 | 0.0165 | ok |  |
| torch_compile | 2 | 128 | 32 | float32 | False | 0.2194 | 0.0571 | 0.6956 | 0.1759 | 0.0475 | 0.0477 | ok |  |
| flash_pytorch | 2 | 128 | 32 | float32 | False | 10.5215 | 0.5026 | 13.1737 | 1.9569 | 0.0161 | 0.0162 | ok |  |
| flash_triton | 2 | 128 | 32 | float32 | False | 0.1105 | 0.0578 | 0.6645 | 0.1948 | 0.0160 | 0.0478 | ok |  |
