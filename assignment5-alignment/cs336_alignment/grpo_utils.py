from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor

from cs336_alignment.sft_utils import masked_mean


# Computes per-rollout rewards and group-normalized GRPO advantages.
def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [reward["reward"] for reward in reward_dicts], dtype=torch.float32
    )
    grouped = raw_rewards.view(-1, group_size)
    advantages = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        advantages = advantages / (grouped.std(dim=1, keepdim=True) + advantage_eps)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": sum(r.get("format_reward", 0.0) for r in reward_dicts)
        / len(reward_dicts),
        "mean_answer_reward": sum(r.get("answer_reward", 0.0) for r in reward_dicts)
        / len(reward_dicts),
    }
    return advantages.reshape(-1), raw_rewards, metadata


# Computes the vanilla per-token policy-gradient loss from rewards or advantages.
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


# Computes the clipped GRPO objective using old-policy log probabilities.
def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    loss = -torch.minimum(unclipped, clipped)
    metadata = {
        "clip_fraction": ((ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)).float()
    }
    return loss, metadata


# Dispatches to the requested policy-gradient loss variant.
def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: str,
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


# Backpropagates one GRPO microbatch loss over response tokens.
def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = masked_mean(per_token_loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata
