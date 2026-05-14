| model_size | mode | precision | batch_size | context_length | vocab_size | d_model | d_ff | num_layers | num_heads | num_parameters_m | mean_ms | std_ms | peak_gib | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | forward | fp32 | 1 | 2048 | 10000 | 768 | 3072 | 12 | 12 | 128.6254 | 320.9344 | 23.7973 | 7.3841 | ok |  |
| small | forward_backward | fp32 | 1 | 2048 | 10000 | 768 | 3072 | 12 | 12 | 128.6254 | 1025.5439 | 19.8327 | 7.8318 | ok |  |
| small | train_step | fp32 | 1 | 2048 | 10000 | 768 | 3072 | 12 | 12 | 128.6254 | 3885.2113 | 59.5054 | 8.7916 | ok |  |
| small | forward | bf16 | 1 | 2048 | 10000 | 768 | 3072 | 12 | 12 | 128.6254 | 167.7038 | 0.4507 | 5.6843 | ok |  |
| small | forward_backward | bf16 | 1 | 2048 | 10000 | 768 | 3072 | 12 | 12 | 128.6254 | 529.6854 | 3.4579 | 6.2789 | ok |  |
| small | train_step | bf16 | 1 | 2048 | 10000 | 768 | 3072 | 12 | 12 | 128.6254 | 1320.7009 | 12.6794 | 7.2438 | ok |  |
