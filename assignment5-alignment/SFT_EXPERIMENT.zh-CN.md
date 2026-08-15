# A5 SFT 实验说明与运行手册

[中文](SFT_EXPERIMENT.zh-CN.md) | [English](SFT_EXPERIMENT.md)

这份文档记录当前 SFT 实验的完整流程：数据如何处理、训练优化目标是什么、奖励和评价指标如何计算，以及服务器上可以直接复制执行的命令。

## 1. 实验目标

当前实验是在 `Qwen2.5-Math-1.5B` base model 上做 GSM8K 的 SFT。

训练阶段使用官方 GSM8K 的标准解题过程作为监督数据，让模型学习输出符合 `r1_zero.prompt` 要求的格式：

```text
<think> reasoning process here </think> <answer> answer here </answer>
```

注意：训练本身不是 RL，不直接使用 reward 做反向传播。reward 只用于定期 evaluation，检查模型生成的答案格式和最终答案是否正确。

## 2. 数据流程

### 2.1 原始数据

原始 GSM8K 数据位于：

```text
data/gsm8k/train.jsonl
data/gsm8k/test.jsonl
```

原始每行格式大致为：

```json
{
  "question": "问题文本",
  "answer": "解题过程 ... #### 最终答案"
}
```

GSM8K 的 `answer` 字段中，`####` 前面是推理过程，`####` 后面是最终答案。

### 2.2 转成 SFT 格式

训练脚本不直接读取原始 GSM8K，而是读取预处理后的 SFT jsonl。转换脚本是：

```text
scripts/prepare_gsm8k_sft.py
```

转换命令：

```bash
python scripts/prepare_gsm8k_sft.py \
  --input_path data/gsm8k/train.jsonl \
  --output_path data/gsm8k/train_sft_r1_zero.jsonl \
  --prompt_path cs336_alignment/prompts/r1_zero.prompt
```

转换后的每行格式为：

```json
{
  "prompt": "r1_zero.prompt 填入 question 后的完整 prompt",
  "response": "GSM8K 解题过程\n</think> <answer>最终答案</answer>",
  "ground_truth": "最终答案"
}
```

这里的空格很重要：

```text
</think> <answer>
```

`r1_zero_reward_fn` 会严格检查这个字符串。如果写成 `</think><answer>`，格式分会失败。

### 2.3 训练时 tokenization

训练代码在启动后一次性把全部训练数据 tokenize：

```python
tokenized_train_data = tokenize_prompt_and_output(...)
```

`tokenize_prompt_and_output` 做三件事：

1. 把 `prompt` 单独 tokenize，用来确定 response 从哪里开始。
2. 把 `prompt + response` 拼起来 tokenize。
3. 构造 `response_mask`，只在 response token 上计算 SFT loss，不训练模型复述 prompt。

实际张量含义：

```text
input_ids      = prompt + response 的 token，去掉最后一个 token
labels         = prompt + response 的 token，去掉第一个 token
response_mask  = labels 中属于 response 的位置为 True
```

这是标准 causal LM next-token prediction 设置。

## 3. 损失函数与优化目标

### 3.1 token log probability

`get_response_log_probs` 会先跑 policy model：

```python
logits = model(input_ids=input_ids).logits
log_probs = F.log_softmax(logits, dim=-1)
```

然后用 `labels` 从 vocabulary 维度取出目标 token 的 log probability：

```python
token_log_probs = gather(log_probs, labels)
```

所以这里得到的是：

```text
log p_theta(label_t | prefix)
```

不是负数 loss。取负号是在 SFT loss 里完成的。

### 3.2 当前代码使用的 SFT loss

当前训练循环调用：

```python
loss, _ = sft_microbatch_train_step(
    policy_log_probs=log_probs,
    response_mask=batch["response_mask"],
    gradient_accumulation_steps=grad_accum_steps,
    normalize_constant=1.0,
)
```

因此当前优化目标是：

```text
loss = - mean_over_batch( sum_over_response_tokens log p_theta(y_t | x, y_<t) )
```

也就是每个样本先把所有 response token 的负 log probability 求和，然后在 batch 维度平均。

这不是 token-level mean，而是 sequence-level sum 后 batch mean。因为 `normalize_constant=1.0`，没有除以 response token 数量。

如果未来把 `normalize_constant=None`，代码会走另一条逻辑：

```text
loss = - mean_over_batch( mean_over_response_tokens log p_theta(y_t | x, y_<t) )
```

也就是每个样本内部先做 token 平均。

### 3.3 gradient accumulation

当前训练是 step-based，不是 epoch-based。

```text
gradient_accumulation_steps = batch_size / micro_batch_size
```

例如：

```text
batch_size = 16
micro_batch_size = 1
gradient_accumulation_steps = 16
```

含义是：

1. 每次只把 1 条样本放进 GPU 做 forward/backward。
2. 连续累积 16 次梯度。
3. 调一次 `optimizer.step()`，这才算完成 1 个 training step。

因此总训练样本数约为：

```text
max_steps * batch_size
```

例如：

```text
200 steps * batch size 16 = 3200 examples
```

注意这里是随机采样，有放回，不保证严格遍历完整数据集。

## 4. 奖励策略

奖励函数是：

```text
cs336_alignment.drgrpo_grader.r1_zero_reward_fn
```

它只在 evaluation 时使用，不参与 SFT 训练梯度。

### 4.1 格式检查

模型输出必须包含：

```text
</think> <answer>
</answer>
```

如果格式不满足：

```text
format_reward = 0
answer_reward = 0
reward = 0
```

### 4.2 答案检查

如果格式正确，会从 `<answer>...</answer>` 中提取最终答案，然后和 ground truth 比较。

比较函数会做一些数学答案归一化和验证，例如整数、分数、LaTeX 表达式等。

如果答案正确：

```text
format_reward = 1
answer_reward = 1
reward = 1
```

如果格式正确但答案错误：

```text
format_reward = 1
answer_reward = 0
reward = 0
```

因此：

- `format_score` 反映模型是否学会输出指定格式。
- `answer_score` 反映格式正确以后，最终答案是否正确。
- `accuracy` 使用总 reward，等价于“格式正确且答案正确”的比例。

## 5. Evaluation 逻辑

训练脚本会在这些时刻评估：

```text
step 0
每隔 eval_every_steps
```

例如 `--eval_every_steps 50`，会在：

```text
0, 50, 100, 150, 200
```

做 evaluation。

评估使用 vLLM，不用训练中的 HF forward，因此训练和评估分开占用两张 GPU：

```text
GPU 0: policy 训练
GPU 1: vLLM 生成评估
```

每次 eval 前，代码会把当前 policy 权重同步到 vLLM：

```python
load_policy_into_vllm_instance(policy, vllm_inst)
```

### 5.1 生成参数

当前 evaluation sampling 参数：

```python
SamplingParams(
    temperature=0.0,
    max_tokens=args.max_tokens,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
```

含义：

- `temperature=0.0`：greedy decoding，减少评估随机性。
- `max_tokens`：每题最多生成多少 token。
- `stop=["</answer>"]`：生成到答案结束标签后停止。
- `include_stop_str_in_output=True`：把 `</answer>` 保留在输出中，方便 reward 检查格式。

### 5.2 W&B 指标

训练指标：

```text
train/loss
train/global_entropy
train/response_entropy
```

评估指标：

```text
eval/accuracy
eval/format_score
eval/answer_score
eval/avg_length
eval/avg_length_correct
eval/avg_length_incorrect
eval/samples
```

指标含义：

- `train/loss`：SFT negative log likelihood，越低通常表示越拟合训练数据。
- `train/global_entropy`：所有有效 token 位置的平均 entropy。
- `train/response_entropy`：response token 位置的平均 entropy。
- `eval/accuracy`：`reward` 平均值，即格式正确且答案正确的比例。
- `eval/format_score`：格式正确比例。
- `eval/answer_score`：最终答案正确比例。
- `eval/avg_length`：平均生成长度。
- `eval/avg_length_correct`：答对样本的平均生成长度。
- `eval/avg_length_incorrect`：答错样本的平均生成长度。
- `eval/samples`：W&B table，用来看具体 prompt、response、ground truth 和 reward。

如果 W&B 上没看到 `eval/answer_score`，先确认：

1. 是否已经跑到 eval step。
2. 图表名字是否是 `eval/answer_score`，不是 `answer_score`。
3. 是否因为磁盘满导致 table artifact 写入失败。标量 metrics 应该优先保留，table 只是辅助查看。

## 6. Checkpoint 策略

当前训练代码只保存最终模型，不保存中间 checkpoint。

最终模型路径为：

```text
result/checkpoints/sft_steps{max_steps}_subset{dataset_size}_filtered{filter_correct}
```

例如正式 200 step 且没有设置 `dataset_size`、没有 `--filter_correct`：

```text
result/checkpoints/sft_steps200_subsetNone_filteredFalse
```

里面应该包含：

```text
model.safetensors
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.json
merges.txt
```

检查命令：

```bash
find result/checkpoints -maxdepth 3 -type f | head -50
```

注意：命令行显式传入的 `--output_dir <local-checkpoint-dir>` 会覆盖代码默认值。若要保存到项目内，请显式指定一个相对目录：

```bash
--output_dir result/checkpoints
```

## 7. 服务器操作流程

### 7.1 进入项目并激活环境

```bash
cd <path-to-assignment5-alignment>
source .venv/bin/activate
```

检查 Python 环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import vllm; print('vllm ok')"
nvidia-smi
```

检查 W&B：

```bash
wandb status
```

如果没有登录：

```bash
wandb login
```

### 7.2 准备数据

```bash
python scripts/prepare_gsm8k_sft.py \
  --input_path data/gsm8k/train.jsonl \
  --output_path data/gsm8k/train_sft_r1_zero.jsonl \
  --prompt_path cs336_alignment/prompts/r1_zero.prompt
```

检查生成文件：

```bash
head -1 data/gsm8k/train_sft_r1_zero.jsonl
```

### 7.3 开 tmux

```bash
tmux new -s sft
```

退出但保持训练继续运行：

```text
Ctrl-b d
```

重新进入：

```bash
tmux attach -t sft
```

### 7.4 Smoke test

正式训练前先跑小规模测试，确认模型、vLLM、W&B、数据格式都没问题。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m cs336_alignment.train_sft_step \
  --model_id models/Qwen2.5-Math-1.5B \
  --train_data_path data/gsm8k/train_sft_r1_zero.jsonl \
  --val_data_path data/gsm8k/test.jsonl \
  --output_dir result/checkpoints \
  --device cuda:0 \
  --vllm_device cuda:1 \
  --vllm_gpu_util 0.65 \
  --dataset_size 8 \
  --max_steps 1 \
  --batch_size 2 \
  --micro_batch_size 1 \
  --max_eval_samples 4 \
  --max_tokens 128 \
  --eval_every_steps 1 \
  --save_every_steps 0 \
  --wandb_project cs336-sft \
  --wandb_run_name smoke-test
```

如果 smoke test 里 `eval/format_score` 和 `eval/answer_score` 能正常出现，就说明主流程基本可跑。

### 7.5 正式训练

当前推荐命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m cs336_alignment.train_sft_step \
  --model_id models/Qwen2.5-Math-1.5B \
  --train_data_path data/gsm8k/train_sft_r1_zero.jsonl \
  --val_data_path data/gsm8k/test.jsonl \
  --output_dir result/checkpoints \
  --device cuda:0 \
  --vllm_device cuda:1 \
  --vllm_gpu_util 0.65 \
  --batch_size 16 \
  --micro_batch_size 1 \
  --max_steps 200 \
  --max_eval_samples 100 \
  --max_tokens 1024 \
  --eval_every_steps 50 \
  --save_every_steps 0 \
  --wandb_project cs336-sft \
  --wandb_run_name qwen2.5-math-1.5b-sft-gsm8k-200steps
```

参数解释：

- `CUDA_VISIBLE_DEVICES=0,1`：让程序看到 2 张 GPU。
- `--device cuda:0`：HF policy model 放在 GPU 0 上训练。
- `--vllm_device cuda:1`：vLLM 放在 GPU 1 上评估。
- `--vllm_gpu_util 0.65`：限制 vLLM 使用 GPU 1 显存比例。
- `--batch_size 16`：一次 optimizer step 的逻辑 batch size。
- `--micro_batch_size 1`：一次 forward/backward 实际进 GPU 的样本数。
- `--max_steps 200`：总 optimizer update 次数。
- `--max_eval_samples 100`：每次评估用 100 道题。
- `--max_tokens 1024`：每道题最多生成 1024 token。
- `--eval_every_steps 50`：每 50 个 step 评估一次。
- `--save_every_steps 0`：当前代码不保存中间 checkpoint，只保存最终模型。
- `--output_dir result/checkpoints`：最终模型保存到项目内的 checkpoint 目录。

## 8. 监控与排查

### 8.1 GPU 监控

另开一个 SSH 窗口：

```bash
watch -n 2 nvidia-smi
```

正常现象：

- GPU 0 训练时利用率高。
- GPU 1 被 vLLM 占显存，但大部分训练时间利用率可能是 0。
- 到 eval step 时，GPU 1 利用率会上升。

### 8.2 磁盘检查

训练前检查空间：

```bash
df -h
du -sh result wandb 2>/dev/null
```

最终 checkpoint 大约需要几 GB 空间。如果 `/` 分区满了，可能出现：

```text
OSError: [Errno 5] Input/output error
```

这种错误经常发生在 W&B table artifact 或 checkpoint 写文件时。先清理空间，再继续跑。

### 8.3 常见错误

如果出现：

```text
ModuleNotFoundError: No module named 'torch'
```

说明没有进入虚拟环境：

```bash
source .venv/bin/activate
```

如果 vLLM OOM：

```bash
--vllm_gpu_util 0.5
```

如果训练 GPU OOM：

```bash
--batch_size 8 --micro_batch_size 1
```

如果 eval accuracy 一直是 0，优先检查：

1. `response` 里是否有 `</think> <answer>`，中间必须有空格。
2. `stop=["</answer>"]` 和 `include_stop_str_in_output=True` 是否保留。
3. W&B 的 `eval/format_score` 是否大于 0。如果格式分是 0，说明主要是格式问题；如果格式分高但答案分低，说明主要是数学答案问题。

## 9. 从服务器拉回本地

本地目标目录：

```text
assignment5-alignment/result/checkpoints
```

在本地仓库根目录执行：

```bash
rsync -avP <remote-user>@<remote-host>:<remote-project-path>/result/checkpoints/<checkpoint-name> assignment5-alignment/result/checkpoints/
```

如果 SSH 有自定义端口：

```bash
rsync -avP -e "ssh -p <port>" <remote-user>@<remote-host>:<remote-project-path>/result/checkpoints/<checkpoint-name> assignment5-alignment/result/checkpoints/
```

拉完检查：

```bash
find assignment5-alignment/result/checkpoints -maxdepth 3 -type f | head -50
```
