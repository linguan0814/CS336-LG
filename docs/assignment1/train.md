# Training Pipeline

This note explains the training system for the Assignment 1 Transformer
language model.

The goal is to understand how tokenized data becomes training batches, how the
model produces logits, how loss and gradients are computed, and how optimizer,
validation, logging, and checkpointing fit together.

The training pipeline is:

```text
tokenized dataset files
  ↓
np.memmap train.dat / valid.dat
  ↓
random batch sampling
  ↓
input_ids / target_ids
  ↓
Transformer LM forward pass
  ↓
next-token cross entropy loss
  ↓
backpropagation
  ↓
gradient clipping
  ↓
AdamW optimizer step
  ↓
logging / validation / checkpointing
```

The implementation lives mainly in:

| File | Purpose |
|---|---|
| `cs336_basics/train.py` | Main training script and training loop |
| `cs336_basics/trainer/data_loading.py` | Randomly samples token sequences from `.dat` files |
| `cs336_basics/trainer/utils.py` | Cross entropy, learning-rate schedule, gradient clipping |
| `cs336_basics/trainer/AdamW.py` | AdamW optimizer implementation |
| `cs336_basics/check_pointing.py` | Save and load training checkpoints |
| `cs336_basics/trainer/experiment_logger.py` | Local JSON/CSV logging and run metadata |
| `cs336_basics/model/transformer.py` | Transformer LM being trained |

Document map:

| Section | Core content |
|---|---|
| [1. What Training Optimizes](#1-what-training-optimizes) | Next-token prediction objective and shifted targets |
| [2. Training Data Flow](#2-training-data-flow) | End-to-end flow from `.dat` token ids to optimizer updates |
| [3. Function and Module Map](#3-function-and-module-map) | Table of training functions and what each does to data/state |
| [4. Configuration and Initialization](#4-configuration-and-initialization) | CLI args, device, metadata, model construction, optimizer, logging |
| [5. Dataset Loading and Batch Sampling](#5-dataset-loading-and-batch-sampling) | `np.memmap`, random start indices, input/target pairs |
| [6. Forward Pass and Cross Entropy Loss](#6-forward-pass-and-cross-entropy-loss) | Logits shape, flattening, cross entropy meaning and implementation |
| [7. Backward Pass and Gradient Handling](#7-backward-pass-and-gradient-handling) | Backprop, non-finite checks, gradient clipping |
| [8. Learning-rate Schedule](#8-learning-rate-schedule) | Warmup, cosine decay, and minimum LR |
| [9. AdamW Optimizer](#9-adamw-optimizer) | Moment estimates, bias correction, decoupled weight decay |
| [10. Logging and Validation](#10-logging-and-validation) | Train metrics, validation loop, perplexity, local/W&B logs |
| [11. Checkpointing and Resume](#11-checkpointing-and-resume) | Model/optimizer state, iteration, model config |
| [12. Important Invariants](#12-important-invariants) | Shape, dtype, vocabulary, and training-loop assumptions |
| [13. Summary](#13-summary) | Compact recap of the training pipeline |

---

## 1. What Training Optimizes

The model is trained with next-token prediction.

The tokenized dataset is one long sequence:

```text
[x0, x1, x2, x3, ..., xn]
```

For a context length `S`, one training example is:

```text
input_ids:  [xi,   xi+1, xi+2, ..., xi+S-1]
target_ids: [xi+1, xi+2, xi+3, ..., xi+S]
```

So every target is the input shifted left by one token.

The model computes:

```text
logits = model(input_ids)
```

with shape:

```text
input_ids: (B, S)
logits:    (B, S, vocab_size)
targets:   (B, S)
```

Cross entropy compares each position's vocabulary logits with the correct next
token id.

The causal mask inside the Transformer makes this objective valid: position
`s` can only use tokens up to position `s`, not the target token at `s + 1`.

---

## 2. Training Data Flow

The main training loop in `train.py` performs this sequence every iteration:

```text
iteration number
  ↓
learning_rate_schedule(...)
  ↓
set optimizer lr
  ↓
data_loading(train_data, batch_size, context_len)
  ↓
input_ids / target_ids
  ↓
optimizer.zero_grad()
  ↓
logits = model(input_ids)
  ↓
flatten logits and targets
  ↓
loss = cross_entropy(logits_flat, targets_flat)
  ↓
loss.backward()
  ↓
check gradients are finite
  ↓
gradient_clipping(...)
  ↓
optimizer.step()
  ↓
log metrics
  ↓
optional validation
  ↓
optional checkpoint
```

Training is step-based, not epoch-based.

The approximate number of training tokens processed is:

```text
tokens_processed = train_steps × batch_size × context_len
```

This is natural for language model pretraining because the dataset is treated
as a long stream of token ids.

---

## 3. Function and Module Map

| Function / object | Input | Operation on data/state | Output |
|---|---|---|---|
| `parse_args()` | CLI arguments | Parses model, optimizer, training, data, and logging configuration | `argparse.Namespace` |
| `get_device(device_arg)` | `"auto"`, `"cpu"`, `"cuda"`, or `"mps"` | Selects available hardware device | device string |
| `load_data_meta(data_dir)` | tokenized data directory | Reads `meta.json` if present | metadata dict |
| `get_model_config(args)` | parsed args | Extracts model hyperparameters used to construct `transformer_lm` | model config dict |
| `transformer_lm(...)` | model config | Builds token embedding, Transformer blocks, final RMSNorm, and LM head | trainable model |
| `AdamW(model.parameters(), ...)` | parameters and optimizer hyperparameters | Creates optimizer state container | optimizer |
| `get_dataset_memmap(path)` | `.dat` file path | Opens token-id file with `np.memmap` | memory-mapped token array |
| `data_loading(dataset, batch_size, context_length, device)` | memmap token array | Samples random contiguous sequences and shifted targets | `(input_ids, target_ids)` |
| `learning_rate_schedule(...)` | iteration and LR schedule config | Computes warmup/cosine/min LR value | scalar learning rate |
| `model(input_ids)` | token ids `(B, S)` | Runs Transformer LM forward pass | logits `(B, S, vocab_size)` |
| `cross_entropy(logits_flat, targets_flat)` | flattened logits and target ids | Computes mean negative log likelihood | scalar loss |
| `loss.backward()` | scalar loss | Backpropagates gradients into model parameters | parameter gradients |
| `gradient_clipping(parameters, max_l2_norm)` | parameter gradients | Scales gradients if global L2 norm is too large | in-place clipped gradients |
| `optimizer.step()` | parameters, gradients, optimizer state | Applies AdamW parameter update | updated model parameters |
| `LocalExperimentLogger.log_metric(record)` | metric dict | Appends metrics to local JSONL/CSV logs | log files |
| `wandb.log(...)` | metric dict | Sends metrics to W&B if enabled | remote run metrics |
| `save_checkpoint(model, optimizer, iteration, out, model_config)` | model/optimizer state | Serializes training state | `.pt` checkpoint |
| `load_checkpoint(src, model, optimizer)` | checkpoint path | Restores model and optimizer state | saved iteration |

---

## 4. Configuration and Initialization

Training starts with CLI arguments.

Main model arguments:

| Argument | Meaning |
|---|---|
| `vocab_size` | Must match tokenizer vocabulary size |
| `d_model` | Transformer hidden-state width |
| `d_ff` | SwiGLU feed-forward hidden dimension |
| `context_len` | Sequence length sampled for training |
| `num_heads` | Number of attention heads |
| `num_layers` | Number of Transformer blocks |
| `rope_theta` | RoPE frequency base |

Main optimizer/training arguments:

| Argument | Meaning |
|---|---|
| `max_lr` | Peak learning rate |
| `min_lr` | Final/minimum learning rate |
| `warm_up_it` | Number of warmup iterations |
| `cosine_it` | Iteration where cosine decay reaches `min_lr` |
| `weight_decay` | AdamW decoupled weight decay |
| `batch_size` | Number of sequences per step |
| `train_steps` | Number of optimizer iterations |
| `val_interval` | Validation frequency |
| `val_batches` | Number of validation batches per validation run |
| `save_intervals` | Checkpoint save frequency |

Initialization flow:

```text
parse args
  ↓
resolve device
  ↓
load tokenized data metadata
  ↓
build model config
  ↓
create checkpoint/log directory
  ↓
initialize local logger
  ↓
optionally initialize W&B
  ↓
initialize Transformer LM
  ↓
initialize AdamW
  ↓
open train.dat and valid.dat with np.memmap
  ↓
optionally load checkpoint
```

The model is moved to the selected device:

```python
model = transformer_lm(...).to(device)
```

The tokenized data is not fully loaded into RAM. It is opened with
`np.memmap`, so array slices are read as needed.

---

### 4.1 Model Construction

The model is created from `model_config`:

```python
model = transformer_lm(
    vocab_size=model_config["vocab_size"],
    context_length=model_config["context_length"],
    num_layers=model_config["num_layers"],
    d_model=model_config["d_model"],
    num_heads=model_config["num_heads"],
    rope_theta=model_config["rope_theta"],
    d_ff=model_config["d_ff"],
).to(device)
```

This call builds the full decoder-only Transformer LM.

The arguments control different parts of the architecture:

| Argument | Used by | Meaning |
|---|---|---|
| `vocab_size` | `Embedding`, LM head | Number of tokenizer ids; embedding rows and output logits |
| `context_length` | RoPE / attention config | Maximum sequence length expected by the model |
| `num_layers` | `nn.ModuleList` | Number of Transformer blocks |
| `d_model` | Embedding, attention, FFN, residual stream | Width of hidden states |
| `num_heads` | multi-head self-attention | Number of attention heads |
| `rope_theta` | RoPE | Frequency base for rotary position embeddings |
| `d_ff` | SwiGLU | Hidden dimension inside the feed-forward network |

Inside `transformer_lm.__init__`, these pieces are created:

```text
token_embedding = Embedding(vocab_size, d_model)

layers = [
    transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,
        theta=rope_theta,
    )
    for _ in range(num_layers)
]

output_norm = RMSNorm(d_model)
output_embedding = Linear(d_model, vocab_size)
```

As a structure:

```text
transformer_lm
├── token_embedding
├── layers[0]
│   ├── ln1
│   ├── attn
│   ├── ln2
│   └── ffn
├── layers[1]
├── ...
├── output_norm
└── output_embedding
```

The model's forward interface is:

```text
input_ids: (batch_size, context_length)
  ↓
model(input_ids)
  ↓
logits: (batch_size, context_length, vocab_size)
```

The call to `.to(device)` moves the model parameters and buffers to the
selected device, such as CPU, CUDA, or MPS:

```text
model parameters on CPU
  ↓
.to("cuda")
  ↓
model parameters on GPU
```

This must match the training batch device. If `input_ids` are on GPU but model
parameters are on CPU, PyTorch will raise a device mismatch error.

After construction, the script counts parameters:

```python
total_params = sum(p.numel() for p in model.parameters())
```

This is logged as model metadata.

Important consistency requirements:

1. `model_config["vocab_size"]` must match the tokenizer vocabulary used to
   create `train.dat` and `valid.dat`.
2. `model_config["context_length"]` should match the sampled `context_len`.
3. `d_model` must be divisible by `num_heads`, because each head uses
   `d_k = d_model // num_heads`.
4. A checkpoint can only be loaded into a model with compatible architecture
   shapes.

---

## 5. Dataset Loading and Batch Sampling

The tokenized dataset files contain only integer token ids:

```text
train.dat
valid.dat
```

They are loaded as:

```python
np.memmap(path, dtype=np.uint16, mode="r")
```

The `data_loading` function samples random start positions.

For each item in the batch:

```text
start_idx = random integer
input_seq  = dataset[start_idx     : start_idx + context_length]
target_seq = dataset[start_idx + 1 : start_idx + context_length + 1]
```

This produces:

```text
inputs:  (batch_size, context_length)
targets: (batch_size, context_length)
```

Example:

```text
dataset:  [10, 11, 12, 13, 14, 15]
start:    1
S:        3

input:    [11, 12, 13]
target:   [12, 13, 14]
```

The target sequence is shifted by one token because the model predicts the
next token at every position.

---

## 6. Forward Pass and Cross Entropy Loss

Each training step converts sampled batches to `long` token ids:

```python
input_ids = input_ids.long().to(device)
target_ids = target_ids.long().to(device)
```

The model forward pass returns:

```text
logits: (B, S, vocab_size)
```

For cross entropy, the code flattens the batch and sequence dimensions:

```python
logits_flat = logits.view(-1, logits.size(-1))
targets_flat = target_ids.view(-1)
```

Shape:

```text
logits_flat:  (B * S, vocab_size)
targets_flat: (B * S)
```

---

### 6.1 What Cross Entropy Means

For each prediction position, the model outputs one logit for every vocabulary
token:

```text
logits[i]: (vocab_size)
target[i]: scalar token id
```

The target is the id of the correct next token.

Cross entropy asks:

```text
How much probability did the model assign to the correct next token?
```

If the model assigns high probability to the correct token, the loss is small.
If it assigns low probability to the correct token, the loss is large.

Conceptually:

```text
probabilities = softmax(logits)
loss = -log(probabilities[target])
```

Example:

```text
target token id = 2
probabilities = [0.05, 0.10, 0.80, 0.05]
loss = -log(0.80)
```

The model returns logits instead of probabilities because logits are more
numerically stable to work with. The cross entropy implementation combines the
softmax and negative log likelihood into one calculation.

---

### 6.2 `cross_entropy` Implementation

The implemented cross entropy is:

```python
get_logit = out_logit.gather(dim=-1, index=target.unsqueeze(-1))
logsumexp = torch.logsumexp(out_logit, dim=-1, keepdim=True)
loss = -get_logit + logsumexp
return loss.mean()
```

Mathematically, for one example:

```text
loss = -log softmax(logits)[target]
     = -logit[target] + logsumexp(logits)
```

Data flow:

```text
out_logit: (B * S, vocab_size)
target:    (B * S)
  ↓
target.unsqueeze(-1): (B * S, 1)
  ↓
gather target logits
  ↓
get_logit: (B * S, 1)
  ↓
logsumexp over vocab dimension
  ↓
logsumexp: (B * S, 1)
  ↓
loss per position = -get_logit + logsumexp
  ↓
mean over all positions
  ↓
scalar loss
```

`torch.logsumexp` is used for numerical stability.

The final loss is the mean over all `B * S` token prediction positions.

The training loop checks:

```python
torch.isfinite(loss)
```

before calling backward. If the loss is `nan` or `inf`, training stops and
writes a final checkpoint/summary for the completed iteration.

---

## 7. Backward Pass and Gradient Handling

After computing loss:

```python
loss.backward()
```

PyTorch computes gradients for all trainable parameters.

The training loop then checks each parameter gradient:

```python
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        stop training
```

This catches exploding or invalid gradients before the optimizer step.

Then gradients are clipped:

```python
gradient_clipping(model.parameters(), args.clip_grad_norm)
```

The implementation computes global L2 norm:

```text
global_norm = sqrt(sum over all gradient elements of grad^2)
```

If:

```text
global_norm > max_l2_norm
```

each gradient is scaled by:

```text
max_l2_norm / (global_norm + eps)
```

This preserves gradient direction but limits update magnitude.

---

## 8. Learning-rate Schedule

The learning rate changes every iteration.

The schedule has three phases:

```text
warmup
  ↓
cosine decay
  ↓
minimum learning rate
```

In code:

```python
lr = learning_rate_schedule(
    iter_num,
    max_lr,
    min_lr,
    warm_up_it,
    cosine_it,
)
```

Warmup:

```text
if it < warmup_iters:
    lr = it * max_lr / warmup_iters
```

Cosine decay:

```text
progress = (it - warmup_iters) / (cosine_iters - warmup_iters) * pi
lr = min_lr + 0.5 * (1 + cos(progress)) * (max_lr - min_lr)
```

After the cosine cycle:

```text
lr = min_lr
```

The training loop writes the current learning rate into every optimizer
parameter group before sampling the batch:

```python
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```

Note: the custom `AdamW` implementation stores the learning rate in the
parameter-group key `alpha`. If the optimizer only reads `alpha`, then updating
`lr` alone does not change the optimizer's effective step size. This is a
useful implementation detail to check when reviewing training behavior.

---

## 9. AdamW Optimizer

AdamW combines adaptive moment estimates with decoupled weight decay.

For each parameter, the optimizer stores:

```text
m: first moment estimate
v: second moment estimate
t: step counter
```

Given gradient `g`, the implementation updates:

```text
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
```

Then applies bias-corrected step size:

```text
alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
```

Parameter update:

```text
p = p - alpha_t * m / (sqrt(v) + eps)
```

Then decoupled weight decay:

```text
p = p - alpha * weight_decay * p
```

The important distinction is that weight decay is applied directly to the
parameter, not added into the gradient.

---

## 10. Logging and Validation

Training logs metrics every `log_intervals` steps.

Training metrics:

| Metric | Meaning |
|---|---|
| `loss` | Current step loss |
| `avg_loss` | Mean of recent training losses |
| `perplexity` | `exp(avg_loss)` |
| `learning_rate` | Current scheduled learning rate |

Perplexity is:

```text
PPL = exp(loss)
```

Validation runs every `val_interval` steps, after step 0:

```python
if iter_num % args.val_interval == 0 and iter_num > 0:
    model.eval()
    with torch.no_grad():
        sample val batches
        compute validation loss
    model.train()
```

Validation uses the same data sampling and loss computation as training, but:

1. gradients are disabled with `torch.no_grad()`;
2. optimizer state is not changed;
3. model is temporarily put into eval mode.

Metrics are written to:

```text
config.json
metrics.jsonl
metrics.csv
summary.json
```

inside the checkpoint/log directory, unless local logging is disabled.

If W&B is enabled, train and validation metrics are also sent to W&B.

---

## 11. Checkpointing and Resume

Checkpoints are saved periodically:

```python
checkpoint_{iter_num}.pt
```

and once at the end:

```python
checkpoint_final_{completed_iter}.pt
```

The checkpoint contains:

```text
model_state
optimizer_state
iteration
model_config
```

`model_state` stores all model parameters and buffers.

`optimizer_state` stores AdamW moment estimates and step counters.

`iteration` tells training where to resume.

`model_config` records the model shape so generation can rebuild a compatible
model later.

Resume flow:

```text
load checkpoint
  ↓
model.load_state_dict(...)
  ↓
optimizer.load_state_dict(...)
  ↓
start_iter = checkpoint["iteration"]
  ↓
continue training from start_iter
```

This means resuming restores both model weights and optimizer momentum state.

---

## 12. Important Invariants

The training pipeline relies on these invariants:

1. `train.dat` and `valid.dat` contain token ids produced by the same tokenizer.
2. `vocab_size` matches the tokenizer vocabulary and the maximum token id in
   the dataset.
3. `context_len + 1` tokens must be available for every sampled sequence.
4. `input_ids` and `target_ids` have shape `(batch_size, context_len)`.
5. `target_ids` is exactly `input_ids` shifted by one token in the original
   dataset stream.
6. The model output shape is `(batch_size, context_len, vocab_size)`.
7. Cross entropy receives flattened logits `(B * S, vocab_size)` and targets
   `(B * S)`.
8. Loss and gradients should be finite before optimizer updates.
9. Checkpoints should save model state, optimizer state, iteration, and model
   config.

---

## 13. Summary

The training system connects the tokenized dataset to the Transformer LM
objective.

The core loop is:

```text
sample input/target token sequences
  ↓
run model forward
  ↓
compute next-token cross entropy
  ↓
backpropagate gradients
  ↓
clip gradients
  ↓
AdamW update
  ↓
log / validate / checkpoint
```

The key ideas are:

- The model trains on token ids, not raw text.
- Targets are inputs shifted by one token.
- Training computes all positions in parallel.
- The causal mask prevents future-token leakage.
- Cross entropy is computed from logits using `logsumexp`.
- The learning rate follows warmup plus cosine decay.
- Gradient clipping and non-finite checks protect training stability.
- Checkpoints store enough state to resume training and later run generation.
