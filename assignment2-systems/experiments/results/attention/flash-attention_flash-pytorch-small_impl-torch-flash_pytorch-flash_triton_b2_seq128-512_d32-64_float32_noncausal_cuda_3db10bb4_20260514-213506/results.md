| implementation | batch_size | seq_len | d_model | dtype | causal | forward_mean_ms | forward_std_ms | backward_mean_ms | backward_std_ms | forward_peak_gib | backward_peak_gib | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | 2 | 128 | 32 | float32 | False | 0.1386 | 0.0567 | 0.5317 | 0.1831 | 0.0083 | 0.0165 | ok |  |
| flash_pytorch | 2 | 128 | 32 | float32 | False | 5.6499 | 1.0217 | 6.7395 | 1.1295 | 0.0161 | 0.0162 | ok |  |
| flash_triton | 2 | 128 | 32 | float32 | False | 0.1056 | 0.0142 | 0.5043 | 0.0591 | 0.0160 | 0.0164 | ok |  |
| torch | 2 | 128 | 64 | float32 | False | 0.3181 | 0.1349 | 0.5353 | 0.1126 | 0.0164 | 0.0167 | ok |  |
| flash_pytorch | 2 | 128 | 64 | float32 | False | 3.4612 | 0.1094 | 5.5971 | 0.2586 | 0.0163 | 0.0165 | ok |  |
| flash_triton | 2 | 128 | 64 | float32 | False | 0.0813 | 0.0187 | 0.4782 | 0.0415 | 0.0162 | 0.0479 | ok |  |
| torch | 2 | 256 | 32 | float32 | False | 0.1447 | 0.0112 | 0.4765 | 0.0623 | 0.0172 | 0.0182 | ok |  |
| flash_pytorch | 2 | 256 | 32 | float32 | False | 12.8562 | 0.2776 | 25.9145 | 10.8951 | 0.0162 | 0.0165 | ok |  |
| flash_triton | 2 | 256 | 32 | float32 | False | 0.0993 | 0.0137 | 0.5676 | 0.1382 | 0.0162 | 0.0494 | ok |  |
| torch | 2 | 256 | 64 | float32 | False | 0.1540 | 0.0115 | 0.4976 | 0.0478 | 0.0175 | 0.0186 | ok |  |
| flash_pytorch | 2 | 256 | 64 | float32 | False | 14.7118 | 5.1624 | 22.8112 | 4.0927 | 0.0166 | 0.0170 | ok |  |
| flash_triton | 2 | 256 | 64 | float32 | False | 0.1001 | 0.0444 | 0.4995 | 0.0725 | 0.0165 | 0.0179 | ok |  |
| torch | 2 | 512 | 32 | float32 | False | 0.1658 | 0.0411 | 0.5110 | 0.1020 | 0.0204 | 0.0244 | ok |  |
| flash_pytorch | 2 | 512 | 32 | float32 | False | 53.5588 | 1.1947 | 89.2681 | 8.6689 | 0.0166 | 0.0170 | ok |  |
| flash_triton | 2 | 512 | 32 | float32 | False | 0.0932 | 0.0041 | 0.6157 | 0.0707 | 0.0165 | 0.0223 | ok |  |
| torch | 2 | 512 | 64 | float32 | False | 0.1794 | 0.0287 | 0.6661 | 0.2120 | 0.0210 | 0.0251 | ok |  |
| flash_pytorch | 2 | 512 | 64 | float32 | False | 52.9284 | 1.2258 | 89.0567 | 9.8085 | 0.0172 | 0.0180 | ok |  |
| flash_triton | 2 | 512 | 64 | float32 | False | 0.0876 | 0.0101 | 0.5467 | 0.0157 | 0.0171 | 0.0230 | ok |  |
