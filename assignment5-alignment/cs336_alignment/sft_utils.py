from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from vllm import LLM, SamplingParams


# Stores already-packed fixed-length SFT examples as a PyTorch dataset.
class PackedSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.examples[index]


# Tokenizes prompt/response pairs and marks response-label positions for SFT loss.
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    prompt_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    prompt_output_ids = [
        tokenizer.encode(prompt + output, add_special_tokens=False)
        for prompt, output in zip(prompt_strs, output_strs)
    ]

    max_len = max(len(ids) for ids in prompt_output_ids) - 1
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    input_ids = torch.full((len(prompt_strs), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(prompt_strs), max_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(prompt_strs), max_len), dtype=torch.bool)

    for row, (p_ids, ids) in enumerate(zip(prompt_ids, prompt_output_ids)):
        row_input = ids[:-1]
        row_labels = ids[1:]
        input_ids[row, : len(row_input)] = torch.tensor(row_input, dtype=torch.long)
        labels[row, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long)
        response_start = max(len(p_ids) - 1, 0)
        response_mask[row, response_start : len(row_labels)] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


# Computes categorical entropy over the vocabulary dimension from next-token logits.
def compute_entropy(logits: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1)


# Scores each provided label token under the model, optionally also returning entropy.
def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    output = {"log_probs": token_log_probs}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)
    return output


# Sums masked tensor values along a dimension and divides by a fixed constant.
def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    mask = mask.to(dtype=tensor.dtype, device=tensor.device)
    return (tensor * mask).sum(dim=dim) / normalize_constant


# Averages tensor values over positions selected by a boolean-like mask.
def masked_mean(tensor: Tensor, mask: Tensor, dim: int | None = None) -> Tensor:
    mask = mask.to(dtype=tensor.dtype, device=tensor.device)
    numerator = (tensor * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim)
    return numerator / denominator  #token维度的loss

# Backpropagates one SFT microbatch loss over response tokens.
def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if normalize_constant is None:#若没有normalize_constant传入，则计算token维度的loss
        microbatch_loss_mean = -masked_mean(policy_log_probs, response_mask, dim=-1).mean()
    else:
        microbatch_loss_mean = -masked_normalize(
                                    policy_log_probs,
                                    response_mask,
                                    dim=-1,
                                    normalize_constant=normalize_constant,
                                ).mean()
    scaled_loss = microbatch_loss_mean / gradient_accumulation_steps
    scaled_loss.backward()

    #第一个元素 scaled_loss 用于测试对比
    #第二个元素 记录未缩放前的microbatch 平均 loss 用于日志
    meatadata = {
        "loss":microbatch_loss_mean.detach()
    }

    return scaled_loss, meatadata


# Packs a JSONL SFT dataset into fixed-length next-token prediction examples.
def get_packed_sft_dataset(
    tokenizer,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    with open(dataset_path) as f:
        examples = [json.loads(line) for line in f]
    if shuffle:
        random.shuffle(examples)

    template_path = Path(__file__).parent / "prompts" / "alpaca_sft.prompt"
    template = template_path.read_text().rstrip()

    token_stream: list[int] = []
    for example in examples:
        text = template.format(
            instruction=example["prompt"],
            response=example["response"],
        )
        token_stream.extend(tokenizer.encode(text, add_special_tokens=True))
        if tokenizer.eos_token_id is not None:
            token_stream.append(tokenizer.eos_token_id)

    packed_examples = []
    for start in range(0, len(token_stream) - seq_length, seq_length):
        chunk = token_stream[start : start + seq_length + 1]
        if len(chunk) < seq_length + 1:
            break
        packed_examples.append(
            {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
        )
    return PackedSFTDataset(packed_examples)


# Builds a DataLoader for one epoch over a dataset.
def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Formats prompt/generation pairs into lightweight rows for logging.
def log_generations(
    vllm_model: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    step: int,
    log_prefix: str = "eval",
):
    """
    让模型生成回答并记录详细的评估指标。
    """

    # 1. 模型生成回答
    # 注意：在调用此函数前，应确保已将最新的 policy 权重加载到了 vLLM 实例中
    outputs = vllm_model.generate(prompts, sampling_params)

    table_data = []

    # 用于统计的数据
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []
    total_reward = 0
    total_format_reward = 0
    total_answer_reward = 0

    # 2. 逐条处理生成结果
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer = ground_truths[i]

        # 计算奖励
        scores = reward_fn(generated_text, gold_answer)

        r = scores.get("reward", 0.0)
        fr = scores.get("format_reward", 0.0)
        ar = scores.get("answer_reward", 0.0)

        # 计算响应长度
        resp_len = len(generated_text)
        all_lengths.append(resp_len)

        if r > 0.5:  # 认为是正确的
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)

        total_reward += r
        total_format_reward += fr
        total_answer_reward += ar

        # 准备存入 wandb Table 的数据（展示前几条即可，防止日志过大）
        if i < 100:
            table_data.append([
                step,
                prompts[i],  # 只取 prompt 结尾部分
                generated_text,
                gold_answer,
                r,
                fr,
                ar,
            ])

    # 3. 计算聚合统计量
    metrics = {
        f"{log_prefix}/accuracy": total_reward / len(prompts),
        f"{log_prefix}/format_score": total_format_reward / len(prompts),
        f"{log_prefix}/answer_score": total_answer_reward / len(prompts),
        f"{log_prefix}/avg_length": np.mean(all_lengths),
        f"{log_prefix}/avg_length_correct": np.mean(correct_lengths) if correct_lengths else 0,
        f"{log_prefix}/avg_length_incorrect": np.mean(incorrect_lengths) if incorrect_lengths else 0,
    }

    # 4. 记录到日志系统
    if wandb.run is not None:
        # 记录表格：方便直接在网页看具体的推理过程
        columns = ["step", "prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
        wandb.log(
            {f"{log_prefix}/samples": wandb.Table(columns=columns, data=table_data)},
            step=step,
        )

        # 记录统计数据
        wandb.log(metrics, step=step)

    print(
        f"Step {step}: "
        f"Accuracy: {metrics[f'{log_prefix}/accuracy']:.4f}, "
        f"Avg Len: {metrics[f'{log_prefix}/avg_length']:.1f}"
    )
    return metrics

    return metrics


# Backward-compatible typo used in the in-progress training script.
stf_microbatch_train_step = sft_microbatch_train_step
