| model_size | mode | precision | batch_size | context_length | vocab_size | d_model | d_ff | num_layers | num_heads | num_parameters_m | mean_ms | std_ms | peak_gib | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medium | forward | fp32 | 1 | 512 | 10000 | 1024 | 4096 | 24 | 16 | 423.1834 | 80.9967 | 7.4954 | 3.7957 | ok |  |
| medium | forward_backward | fp32 | 1 | 512 | 10000 | 1024 | 4096 | 24 | 16 | 423.1834 | 251.7277 | 1.2542 | 3.8661 | ok |  |
| medium | train_step | fp32 | 1 | 512 | 10000 | 1024 | 4096 | 24 | 16 | 423.1834 | 2587.2741 | 163.4782 | 7.9076 | ok |  |
| medium | forward | bf16 | 1 | 512 | 10000 | 1024 | 4096 | 24 | 16 | 423.1834 | 105.3974 | 1.1015 | 3.8554 | ok |  |
| medium | forward_backward | bf16 | 1 | 512 | 10000 | 1024 | 4096 | 24 | 16 | 423.1834 | 178.8693 | 9.1846 | 3.8925 | ok |  |
| medium | train_step | bf16 | 1 | 512 | 10000 | 1024 | 4096 | 24 | 16 | 423.1834 | 1797.2398 | 10.7498 | 7.9076 | ok |  |
