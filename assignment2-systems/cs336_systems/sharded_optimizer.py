from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist


class ShardedOptimizer:
    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self.params = self._unique_parameters(params)

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.local_params = [param for idx, param in enumerate(self.params) if self._owner(idx) == self.rank]
        self.optimizer = optimizer_cls(self.local_params, **kwargs) if self.local_params else None

    def zero_grad(self, set_to_none: bool = False):
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    def step(self, closure=None):
        loss = None
        if self.optimizer is not None:
            loss = self.optimizer.step(closure=closure)

        if self.world_size > 1:
            for idx, param in enumerate(self.params):
                dist.broadcast(param.data, src=self._owner(idx))

        return loss

    def state_dict(self):
        if self.optimizer is None:
            return {}
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict)

    def _owner(self, idx: int) -> int:
        return idx % self.world_size

    @staticmethod
    def _unique_parameters(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        unique_params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for param in params:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            unique_params.append(param)
        return unique_params
