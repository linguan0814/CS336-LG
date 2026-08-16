# A5 SFT experiment notes

[中文](SFT_EXPERIMENT.zh-CN.md) | English

This document is the English entry point for the detailed SFT runbook. The complete step-by-step operational notes remain in [the Chinese version](SFT_EXPERIMENT.zh-CN.md), including data preparation, response-only loss, evaluation metrics, checkpoint handling, and local/remote execution examples.

## Scope

The SFT track uses GSM8K-style data and is separate from the Math12K GRPO evidence presented on the project homepage. It covers prompt formatting, response-only supervised loss, evaluation of format/answer scores, and checkpoint management.

## Public-release boundary

All model paths, datasets, checkpoints, W&B logs, and remote-machine details are placeholders or local-only. The runbook documents how the existing experiment was organized; reproducing the original run requires obtaining the same external inputs independently.
