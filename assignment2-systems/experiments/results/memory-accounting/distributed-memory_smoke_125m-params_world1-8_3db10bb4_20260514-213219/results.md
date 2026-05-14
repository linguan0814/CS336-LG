| strategy | world_size | parameters_m | param_gib_per_rank | grad_gib_per_rank | optimizer_state_gib_per_rank | total_gib_per_rank | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DDP | 1 | 125.0000 | 0.4657 | 0.4657 | 0.9313 | 1.8626 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 1 | 125.0000 | 0.4657 | 0.4657 | 0.9313 | 1.8626 | Replicates model and gradients; shards optimizer state. |
| FSDP | 1 | 125.0000 | 0.4657 | 0.4657 | 0.9313 | 1.8626 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
| DDP | 2 | 125.0000 | 0.4657 | 0.4657 | 0.9313 | 1.8626 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 2 | 125.0000 | 0.4657 | 0.4657 | 0.4657 | 1.3970 | Replicates model and gradients; shards optimizer state. |
| FSDP | 2 | 125.0000 | 0.2328 | 0.2328 | 0.4657 | 0.9313 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
| DDP | 4 | 125.0000 | 0.4657 | 0.4657 | 0.9313 | 1.8626 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 4 | 125.0000 | 0.4657 | 0.4657 | 0.2328 | 1.1642 | Replicates model and gradients; shards optimizer state. |
| FSDP | 4 | 125.0000 | 0.1164 | 0.1164 | 0.2328 | 0.4657 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
| DDP | 8 | 125.0000 | 0.4657 | 0.4657 | 0.9313 | 1.8626 | Replicates parameters, gradients, and AdamW first/second moments. |
| ShardedOptimizer | 8 | 125.0000 | 0.4657 | 0.4657 | 0.1164 | 1.0477 | Replicates model and gradients; shards optimizer state. |
| FSDP | 8 | 125.0000 | 0.0582 | 0.0582 | 0.1164 | 0.2328 | Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers. |
