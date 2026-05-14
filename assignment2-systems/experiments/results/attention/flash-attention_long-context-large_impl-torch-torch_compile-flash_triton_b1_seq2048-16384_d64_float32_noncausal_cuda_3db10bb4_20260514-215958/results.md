| implementation | batch_size | seq_len | d_model | dtype | causal | forward_mean_ms | forward_std_ms | backward_mean_ms | backward_std_ms | forward_peak_gib | backward_peak_gib | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | 1 | 2048 | 64 | float32 | False | 0.3796 | 0.0175 | 1.1976 | 0.0308 | 0.0416 | 0.0813 | ok |  |
| torch_compile | 1 | 2048 | 64 | float32 | False | 0.3547 | 0.0106 | 0.8476 | 0.0282 | 0.0339 | 0.0500 | ok |  |
| flash_triton | 1 | 2048 | 64 | float32 | False | 0.2636 | 0.0144 | 0.9876 | 0.0234 | 0.0183 | 0.1121 | ok |  |
| torch | 1 | 4096 | 64 | float32 | False | 2.0043 | 0.0710 | 5.7266 | 0.1150 | 0.1458 | 0.2717 | ok |  |
| torch_compile | 1 | 4096 | 64 | float32 | False | 1.3869 | 0.2192 | 3.4256 | 0.1104 | 0.0833 | 0.1467 | ok |  |
| flash_triton | 1 | 4096 | 64 | float32 | False | 0.4411 | 0.0018 | 3.3217 | 0.1862 | 0.0208 | 0.2395 | ok |  |
| torch | 1 | 8192 | 64 | float32 | False | 7.4675 | 0.2357 | 22.0100 | 0.3632 | 0.5256 | 1.0276 | ok |  |
| torch_compile | 1 | 8192 | 64 | float32 | False | 5.0347 | 0.1789 | 13.4916 | 0.2829 | 0.2756 | 0.5276 | ok |  |
| flash_triton | 1 | 8192 | 64 | float32 | False | 1.6216 | 0.1690 | 12.7996 | 0.3626 | 0.0257 | 0.7757 | ok |  |
| torch | 1 | 16384 | 64 | float32 | False | 29.2567 | 0.2873 | 88.5550 | 2.3209 | 2.0354 | 4.0393 | ok |  |
| torch_compile | 1 | 16384 | 64 | float32 | False | 20.0148 | 0.2112 | 53.3167 | 0.1876 | 1.0354 | 2.0393 | ok |  |
| flash_triton | 1 | 16384 | 64 | float32 | False | 5.4771 | 0.2671 | 49.2892 | 0.5925 | 0.0355 | 3.0355 | ok |  |
