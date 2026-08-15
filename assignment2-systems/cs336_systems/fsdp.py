from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class _ShardedParameter:
    name: str
    module: torch.nn.Module
    parameter: torch.nn.Parameter
    start: int
    end: int
    full_shape: torch.Size


class FullyShardedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self._sharded_by_parameter_id: dict[int, _ShardedParameter] = {}
        self._sharded_by_name: dict[str, _ShardedParameter] = {}
        self._saved_local_shards: dict[int, torch.Tensor] = {}

        self._shard_large_module_weights()
        self._register_replicated_gradient_hooks()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        pass

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params: dict[str, torch.Tensor] = {}
        for name, parameter in self.module.named_parameters():
            sharded = self._sharded_by_name.get(name)
            if sharded is None:
                full_params[name] = parameter.data.detach().clone()
            else:
                full_params[name] = self._all_gather_shards(parameter.data).view(sharded.full_shape)
        return full_params

    def _shard_large_module_weights(self):
        from cs336_basics.model import Embedding, Linear

        named_modules = dict(self.module.named_modules())
        named_parameters = dict(self.module.named_parameters())

        for module_name, child_module in named_modules.items():
            if not isinstance(child_module, (Linear, Embedding)):
                continue
            parameter_name = f"{module_name}.weight" if module_name else "weight"
            parameter = named_parameters[parameter_name]
            full_shape = parameter.data.shape
            assert full_shape[0] % self.world_size == 0
            rows_per_rank = full_shape[0] // self.world_size
            start = self.rank * rows_per_rank
            end = start + rows_per_rank

            if self.world_size > 1:
                dist.broadcast(parameter.data, src=0)

            parameter.data = parameter.data[start:end].contiguous()

            sharded = _ShardedParameter(
                name=parameter_name,
                module=child_module,
                parameter=parameter,
                start=start,
                end=end,
                full_shape=full_shape,
            )
            self._sharded_by_parameter_id[id(parameter)] = sharded
            self._sharded_by_name[parameter_name] = sharded

            child_module.register_forward_pre_hook(self._make_forward_pre_hook(sharded))
            parameter.register_post_accumulate_grad_hook(self._make_sharded_gradient_hook(sharded))

    def _register_replicated_gradient_hooks(self):
        for parameter in self.module.parameters():
            if not parameter.requires_grad or id(parameter) in self._sharded_by_parameter_id:
                continue
            parameter.register_post_accumulate_grad_hook(self._make_replicated_gradient_hook(parameter))

    def _make_forward_pre_hook(self, sharded: _ShardedParameter):
        def hook(module: torch.nn.Module, inputs):
            parameter = sharded.parameter
            self._saved_local_shards[id(parameter)] = parameter.data
            full_parameter = self._all_gather_shards(parameter.data).view(sharded.full_shape)
            if self.compute_dtype is not None:
                full_parameter = full_parameter.to(self.compute_dtype)
            parameter.data = full_parameter
            parameter.grad = None

        return hook

    def _make_sharded_gradient_hook(self, sharded: _ShardedParameter):
        def hook(parameter: torch.nn.Parameter):
            if parameter.grad is None:
                return

            full_grad = parameter.grad.to(torch.float32)
            local_shard = self._saved_local_shards.pop(id(parameter))
            if self.world_size > 1:
                local_grad = torch.empty_like(local_shard, dtype=full_grad.dtype)
                dist.reduce_scatter_tensor(local_grad, full_grad.contiguous(), op=dist.ReduceOp.SUM)
                local_grad.div_(self.world_size)
            else:
                local_grad = full_grad[sharded.start : sharded.end].contiguous()
            parameter.data = local_shard
            parameter.grad = local_grad.to(local_shard.dtype)

        return hook

    def _make_replicated_gradient_hook(self, parameter: torch.nn.Parameter):
        def hook(_: torch.nn.Parameter):
            if parameter.grad is None:
                return
            if self.world_size > 1:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)

        return hook

    def _all_gather_shards(self, local_shard: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return local_shard.detach().clone()

        gathered = [torch.empty_like(local_shard) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_shard.contiguous())
        return torch.cat(gathered, dim=0)
