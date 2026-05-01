from curses import noecho

import torch
from torch import nn
import math
from collections.abc import Iterable

def cross_entropy(out_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    out_logit: (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
                unnormalized logit of jth class for the ith example.
                
    targets: (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
    '''
    get_logit = out_logit.gather(dim=-1, index=target.unsqueeze(-1))#得到target ID对应的概率
    logsumexp = torch.logsumexp(input=out_logit, dim=-1, keepdim=True)#防止上溢，消除了小量（下溢）
    loss = -get_logit + logsumexp #负对数似然和 化简(batch_size，1)
    return loss.mean()

def gradient_clipping(parameters:Iterable[torch.nn.Parameter], max_l2_norm:float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    L2_norm = 0.0
    for g in grads:
        L2_norm += (g.data**2).sum()
    L2_norm = torch.sqrt(L2_norm)
    if L2_norm < max_l2_norm:
        pass
    else:
        for g in grads:
            g.data *= max_l2_norm/(L2_norm+eps)


def learning_rate_schedule(
        it: int,
        max_learning_rate:float,
        min_learning_rate:float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    '''
        Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

        Returns:
            Learning rate at the given iteration under the specified schedule.
    '''
    assert cosine_cycle_iters>warmup_iters, 'Invalid input for iteration striction'
    if it < warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(...)) * (max_learning_rate - min_learning_rate)
        #math.cos((it - warmup_iters)*math.pi/(cosine_cycle_iters - warmup_iters))
    else:
        return min_learning_rate