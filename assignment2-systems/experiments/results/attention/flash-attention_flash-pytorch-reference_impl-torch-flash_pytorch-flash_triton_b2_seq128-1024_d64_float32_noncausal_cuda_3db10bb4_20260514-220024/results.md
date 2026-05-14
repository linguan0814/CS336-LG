| implementation | batch_size | seq_len | d_model | dtype | causal | forward_mean_ms | forward_std_ms | backward_mean_ms | backward_std_ms | forward_peak_gib | backward_peak_gib | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | 2 | 128 | 64 | float32 | False | 0.1588 | 0.0730 | 0.5242 | 0.0949 | 0.0085 | 0.0167 | ok |  |
| flash_pytorch | 2 | 128 | 64 | float32 | False | 3.1270 | 0.2457 | 8.8281 | 1.3954 | 0.0163 | 0.0165 | ok |  |
| flash_triton | 2 | 128 | 64 | float32 | False | 0.0601 | 0.0028 | 0.4537 | 0.0308 | 0.0162 | 0.0479 | ok |  |
| torch | 2 | 256 | 64 | float32 | False | 0.1254 | 0.0116 | 0.4266 | 0.0238 | 0.0175 | 0.0186 | ok |  |
| flash_pytorch | 2 | 256 | 64 | float32 | False | 13.7549 | 0.8030 | 21.0221 | 3.6966 | 0.0166 | 0.0170 | ok |  |
| flash_triton | 2 | 256 | 64 | float32 | False | 0.0743 | 0.0103 | 0.4591 | 0.0375 | 0.0165 | 0.0497 | ok |  |
| torch | 2 | 512 | 64 | float32 | False | 0.1898 | 0.1049 | 0.4459 | 0.0163 | 0.0210 | 0.0251 | ok |  |
| flash_pytorch | 2 | 512 | 64 | float32 | False | 59.8754 | 9.4681 | 137.4298 | 41.3260 | 0.0172 | 0.0180 | ok |  |
| flash_triton | 2 | 512 | 64 | float32 | False | 0.1003 | 0.0037 | 0.5338 | 0.0622 | 0.0171 | 0.0230 | ok |  |
| torch | 2 | 1024 | 64 | float32 | False | 0.2648 | 0.0045 | 0.8582 | 0.0495 | 0.0339 | 0.0500 | ok |  |
| flash_pytorch | 2 | 1024 | 64 | float32 | False | 232.4157 | 28.6016 | 416.5769 | 112.3829 | 0.0184 | 0.0201 | ok |  |
| flash_triton | 2 | 1024 | 64 | float32 | False | 0.2491 | 0.1189 | 0.8712 | 0.1237 | 0.0183 | 0.0418 | ok |  |
