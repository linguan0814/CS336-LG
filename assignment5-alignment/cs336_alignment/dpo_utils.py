from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from cs336_alignment.sft_utils import get_response_log_probs, tokenize_prompt_and_output


# Sums response-token log probabilities for one tokenized prompt/response pair.
def _response_log_prob_sum(model, input_ids: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
    return (log_probs * mask.to(log_probs.dtype)).sum()


# Computes the scalar DPO loss for one chosen/rejected response pair.
def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    chosen = tokenize_prompt_and_output([prompt], [response_chosen], tokenizer)
    rejected = tokenize_prompt_and_output([prompt], [response_rejected], tokenizer)

    pi_chosen = _response_log_prob_sum(
        lm, chosen["input_ids"], chosen["labels"], chosen["response_mask"]
    )
    pi_rejected = _response_log_prob_sum(
        lm, rejected["input_ids"], rejected["labels"], rejected["response_mask"]
    )
    with torch.no_grad():
        ref_chosen = _response_log_prob_sum(
            lm_ref, chosen["input_ids"], chosen["labels"], chosen["response_mask"]
        )
        ref_rejected = _response_log_prob_sum(
            lm_ref, rejected["input_ids"], rejected["labels"], rejected["response_mask"]
        )

    logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(logits)
