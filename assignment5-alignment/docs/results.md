# Selected GRPO results

[中文](results.zh-CN.md) | English

This page records the two controlled Math12K GRPO ablations featured in the repository README. It is a compact, anonymized derivative of the completed local experiment records; no raw W&B links, checkpoints, datasets, or sample generations are included.

## Shared setting

Both studies use the same recorded base model family (`Qwen2.5-Math-1.5B`), Math12K train/test split, rule-based format and answer rewards, group size 8, learning rate `3e-5`, seed 42, 200 GRPO steps, `train_batch_size=256`, `rollout_batch_size=256`, one optimization epoch per rollout batch, and a maximum response length of 1024 tokens. Results below are terminal evaluation metrics from one completed run per condition.

## 1. Advantage standard-deviation normalization

The only intended variable is `use_std_normalization`; both conditions use clipped GRPO and `mask_normalize` length normalization.

| Condition | Accuracy | Format score | Mean response length | Correct-response length | Incorrect-response length |
| --- | ---: | ---: | ---: | ---: | ---: |
| With group std normalization | 48.4% | 84.6% | 730 | 504 | 943 |
| Without group std normalization | **61.4%** | **87.6%** | 963 | 759 | 1,287 |
| Difference (no-std − with-std) | **+13.0 pts** | **+3.0 pts** | +31.8% | +50.7% | +36.4% |

Removing group standard-deviation normalization improved the recorded terminal accuracy, but it also increased the length of both correct and incorrect responses. This is a quality/verbosity trade-off, not evidence that no-std universally improves GRPO.

<p align="center">
  <img src="assets/grpo_std_accuracy.png" alt="GRPO standard-deviation normalization ablation: evaluation accuracy" width="48%">
  <img src="assets/grpo_std_response_length.png" alt="GRPO standard-deviation normalization ablation: average response length" width="48%">
</p>

## 2. DAPO-style length normalization

The intended variable is `length_norm_type`; both conditions use clipped GRPO with group standard-deviation normalization.

| Condition | Accuracy | Format score | Mean response length | Correct-response length | Incorrect-response length |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mask_normalize` | 39.0% | 83.2% | 683 | 379 | 878 |
| `mask_dapo` | **48.2%** | **93.4%** | **623** | **495** | **743** |
| Difference (DAPO − mask) | **+9.2 pts** | **+10.2 pts** | **−8.8%** | +30.7% | **−15.5%** |

In this completed ablation, the DAPO-style normalization allocated more generated tokens to correct responses while reducing the average length of incorrect responses and improving terminal accuracy. The result is one seed and one training budget, so it should be treated as a targeted engineering observation rather than a general claim about all models or datasets.

<p align="center">
  <img src="assets/grpo_dapo_accuracy.png" alt="GRPO length-normalization ablation: evaluation accuracy" width="48%">
  <img src="assets/grpo_dapo_response_length.png" alt="GRPO length-normalization ablation: average response length" width="48%">
</p>

## Scope and exclusions

The repository also includes SFT and Expert Iteration experiments on GSM8K, plus GRPO learning-rate and loss-function diagnostics. They use different experiment tracks or contain stability outcomes that require more context, so they are not used for direct SFT-versus-GRPO claims here. In particular, this page does not claim that the GSM8K SFT results are a baseline for the Math12K GRPO results.
