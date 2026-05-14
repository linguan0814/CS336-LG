| implementation | batch_size | seq_len | d_model | dtype | causal | forward_mean_ms | forward_std_ms | backward_mean_ms | backward_std_ms | forward_peak_gib | backward_peak_gib | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | 1 | 16 | 8 | float32 | False | 0.0622 | 0.0000 | 0.1549 | 0.0000 |  |  | ok |  |
| flash_pytorch | 1 | 16 | 8 | float32 | False | 0.2132 | 0.0000 | 0.2741 | 0.0000 |  |  | ok |  |
