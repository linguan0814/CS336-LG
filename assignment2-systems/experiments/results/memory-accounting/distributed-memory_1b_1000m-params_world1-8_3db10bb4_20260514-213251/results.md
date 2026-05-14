| strategy | world_size | parameters_m | param_gib_per_rank | grad_gib_per_rank | optimizer_state_gib_per_rank | total_gib_per_rank | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DDP | 1 | 1000.0000 | 3.7253 | 3.7253 | 7.4506 | 14.9012 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 1 | 1000.0000 | 3.7253 | 3.7253 | 7.4506 | 14.9012 | Replicates model and gradients; shards optimizer state. |
| FSDP | 1 | 1000.0000 | 3.7253 | 3.7253 | 7.4506 | 14.9012 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
| DDP | 2 | 1000.0000 | 3.7253 | 3.7253 | 7.4506 | 14.9012 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 2 | 1000.0000 | 3.7253 | 3.7253 | 3.7253 | 11.1759 | Replicates model and gradients; shards optimizer state. |
| FSDP | 2 | 1000.0000 | 1.8626 | 1.8626 | 3.7253 | 7.4506 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
| DDP | 4 | 1000.0000 | 3.7253 | 3.7253 | 7.4506 | 14.9012 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 4 | 1000.0000 | 3.7253 | 3.7253 | 1.8626 | 9.3132 | Replicates model and gradients; shards optimizer state. |
| FSDP | 4 | 1000.0000 | 0.9313 | 0.9313 | 1.8626 | 3.7253 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
| DDP | 8 | 1000.0000 | 3.7253 | 3.7253 | 7.4506 | 14.9012 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 8 | 1000.0000 | 3.7253 | 3.7253 | 0.9313 | 8.3819 | Replicates model and gradients; shards optimizer state. |
| FSDP | 8 | 1000.0000 | 0.4657 | 0.4657 | 0.9313 | 1.8626 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
