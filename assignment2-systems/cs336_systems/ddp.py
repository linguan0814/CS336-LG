from __future__ import annotations

import torch
import torch.distributed as dist


class DistributedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._handles: list[dist.Work] = []

        self._distributed = dist.is_available() and dist.is_initialized()
        if self._distributed:
            for parameter in self._unique_parameters():
                dist.broadcast(parameter.data, src=0)

        for parameter in self._unique_parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self._make_gradient_hook(parameter))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

    def _make_gradient_hook(self, parameter: torch.nn.Parameter):
        def hook(_: torch.nn.Parameter):
            if parameter.grad is None:
                return
            if self._distributed:
                parameter.grad.div_(dist.get_world_size())
                handle = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
                self._handles.append(handle)

        return hook

    def _unique_parameters(self):
        seen: set[int] = set()
        for parameter in self.module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            yield parameter
