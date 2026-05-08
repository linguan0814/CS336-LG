# Assignment 1: Transformer Language Model from Scratch

This document summarizes the first milestone of my CS336 LLM systems study project.

Assignment 1 implements a small GPT-style language model pipeline from scratch, including tokenizer training, dataset tokenization, decoder-only Transformer modeling, training, checkpointing, validation, and autoregressive generation.

The key idea of this milestone is to understand how raw text becomes token ids, how token ids are used to train a language model, and how a trained model generates text from a prompt.

---

## 1. Overview

Assignment 1 can be understood as two connected data flows around the same decoder-only Transformer language model.

The first flow is the **training flow**:

```text
Raw text dataset
    ↓
Train byte-level BPE tokenizer
    ↓
vocab / merges
    ↓
Tokenize dataset into token ids
    ↓
train.dat / valid.dat
    ↓
Random sequence sampling
    ↓
Decoder-only Transformer LM
    ↓
Next-token prediction loss
    ↓
AdamW optimization
    ↓
Checkpointed trained model
```

The second flow is the **generation flow**:

```text
Prompt text
    ↓
tokenizer.encode(prompt)
    ↓
input token ids
    ↓
trained decoder-only Transformer LM
    ↓
take logits at the last position
    ↓
temperature / top-p sampling
    ↓
append sampled token id
    ↓
repeat
    ↓
tokenizer.decode(token ids)
    ↓
generated text
```




Training and generation use the same model, but they run in different ways.

During training, the model predicts the next token for every position in the input sequence in parallel.

During generation, the model samples one token at a time and appends it back to the context.


```text
                         ┌──────────────────────┐
Raw text dataset ───────▶│ Train BPE tokenizer   │
                         └──────────┬───────────┘
                                    │
                           vocab.pkl / merges.pkl
                                    │
                                    ▼
                         ┌──────────────────────┐
Raw text dataset ───────▶│ Tokenize dataset      │
                         └──────────┬───────────┘
                                    │
                           train.dat / valid.dat
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Train Transformer LM  │
                         └──────────┬───────────┘
                                    │
                              checkpoint.pt
                                    │
                                    ▼
Prompt ──▶ tokenizer.encode ──▶ Transformer LM ──▶ sampling ──▶ tokenizer.decode ──▶ text
```

---

## 2. What This Milestone Implements

The assignment covers the full small-scale language model pipeline.

Implemented components:

| Module | Purpose |
|---|---|
| Byte-level BPE tokenizer | Convert raw text into token ids |
| Dataset tokenization | Convert large text files into `train.dat` and `valid.dat` |
| Decoder-only Transformer LM | Predict next-token distributions |
| RMSNorm | Normalize hidden states |
| RoPE | Inject positional information into attention |
| Causal multi-head self-attention | Prevent tokens from attending to future tokens |
| SwiGLU feed-forward network | Modern Transformer FFN block |
| Cross entropy loss | Train next-token prediction |
| AdamW optimizer | Update model parameters |
| Learning-rate schedule | Warmup + cosine decay |
| Gradient clipping | Stabilize training |
| Checkpointing | Save and resume training |
| Validation loop | Monitor validation loss and perplexity |
| Generation script | Generate text from a trained checkpoint |

---

## 3. Repository Structure

Relevant project structure:

```text
assignment1-basics/
├── cs336_basics/
│   ├── tokenizer.py
│   ├── check_pointing.py
│   ├── model/
│   │   └── transformer.py
│   ├── trainer/
│   │   ├── AdamW.py
│   │   ├── data_loading.py
│   │   └── utils.py
│   └── scripts/
│       ├── train_bpe.py
│       ├── tokenize_dataset.py
│       └── generate.py
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_optimizer.py
│   ├── test_train_bpe.py
│   └── ...
└── train.py
```

Documentation structure:

```text
docs/
└── assignment1/
    ├── README.md
    ├── BPE-tokenizer.md
    ├── transformer_lm.md
    ├── training_pipeline.md
    ├── generation.md
    ├── experiments.md
    └── debugging_notes.md
```

---

## 4. Training Flow

The training flow starts from raw text and ends with a trained checkpoint.

### 4.1 Raw Text Dataset

The raw dataset is plain text.

Example datasets:

| Dataset | Purpose |
|---|---|
| TinyStories | Small-scale language modeling and debugging |
| OpenWebText | Larger and more realistic web-text pretraining |

Raw text cannot be directly used by the Transformer. It must first be converted into integer token ids.

---

### 4.2 Train Byte-level BPE Tokenizer

The tokenizer maps text into token ids.

The tokenizer training process learns:

```text
vocab:  token id → byte sequence
merges: ordered BPE merge rules
```

For TinyStories, the tokenizer uses:

```text
vocab_size = 10000
special token = <|endoftext|>
```

The special token is used to mark document boundaries.

Key points:

- The tokenizer is byte-level, so it can represent arbitrary Unicode text.
- BPE merges frequent byte sequences into larger subword tokens.
- The final tokenizer is saved as vocabulary and merge files.
- The tokenizer is trained before the language model is trained.

Tokenizer artifacts:

```text
vocab.pkl
merges.pkl
meta.json
```

---

### 4.3 Tokenize the Dataset

After tokenizer training, the raw dataset is encoded into token ids.

Input:

```text
TinyStoriesV2-GPT4-train.txt
TinyStoriesV2-GPT4-valid.txt
```

Output:

```text
train.dat
valid.dat
```

The `.dat` files store only integer token ids.

They do not store:

```text
token strings
token bytes
token positions
raw text
```

For TinyStories with `vocab_size = 10000`, `uint16` is enough:

```text
max uint16 value = 65535
vocab size = 10000
```

Therefore, the tokenized dataset can be saved compactly as `uint16`.

The training script reads the tokenized data using `np.memmap`, which avoids loading the entire dataset into RAM.

---

### 4.4 Random Sequence Sampling

The tokenized dataset is treated as one long token sequence:

```text
[x0, x1, x2, x3, x4, ..., xn]
```

Each training step randomly samples subsequences.

For a context length of `S`, one training example looks like this:

```text
input_ids:   [xi,   xi+1, xi+2, ..., xi+S-1]
target_ids:  [xi+1, xi+2, xi+3, ..., xi+S]
```

The target sequence is the input sequence shifted by one token.

This is the core data format for next-token prediction.

---

### 4.5 Decoder-only Transformer Training

The model receives token ids:

```text
input_ids: (batch_size, sequence_length)
```

The model outputs logits:

```text
logits: (batch_size, sequence_length, vocab_size)
```

Each position predicts the next token.

Example:

```text
tokens:      [x0, x1, x2, x3, x4]
input_ids:   [x0, x1, x2, x3]
target_ids:  [x1, x2, x3, x4]
```

The model learns:

```text
x0             → x1
x0, x1         → x2
x0, x1, x2     → x3
x0, x1, x2, x3 → x4
```

This is done in one forward pass because causal masking prevents each position from seeing future tokens.

Training does not call `generate`.

Training uses teacher forcing:

```text
known input sequence
    ↓
parallel next-token prediction
    ↓
cross entropy loss
    ↓
backpropagation
    ↓
optimizer step
```

---

## 5. Model Architecture

The model is a decoder-only Transformer language model.

Model interface:

```text
input_ids: (B, S)
logits:    (B, S, vocab_size)
```

Where:

```text
B = batch size
S = sequence length / context length
```

Architecture:

```text
token ids
    ↓
token embedding
    ↓
Transformer block × N
    ↓
final RMSNorm
    ↓
LM head
    ↓
logits over vocabulary
```

Each Transformer block contains:

```text
input
    ↓
RMSNorm
    ↓
causal multi-head self-attention with RoPE
    ↓
residual connection
    ↓
RMSNorm
    ↓
SwiGLU feed-forward network
    ↓
residual connection
    ↓
output
```

Main components:

| Component | Role |
|---|---|
| Token embedding | Converts token ids to dense vectors |
| RMSNorm | Stabilizes hidden-state scale |
| Causal self-attention | Allows each token to attend only to past tokens |
| RoPE | Adds positional information to Q/K vectors |
| SwiGLU | Nonlinear feed-forward transformation |
| LM head | Projects hidden states to vocabulary logits |

Important tensor shapes:

```text
input_ids:      (B, S)
embedding:      (B, S, d_model)
Q/K/V:          (B, num_heads, S, d_head)
attention:      (B, num_heads, S, S)
hidden states:  (B, S, d_model)
logits:         (B, S, vocab_size)
```

---

## 6. RoPE and Token Positions

RoPE needs token positions, but positions are not produced by the tokenizer.

Correct responsibility split:

```text
tokenizer:
    text → token ids

tokenized dataset:
    stores token ids only

model.forward:
    creates token positions from sequence length

RoPE:
    uses token positions to rotate Q/K vectors
```

For normal training, if the input shape is:

```text
input_ids: (B, S)
```

then the default positions are:

```text
[0, 1, 2, ..., S-1]
```

These positions can be generated inside `model.forward`.

The tokenized dataset should not store positions.

For future KV-cache inference, positions must account for the number of past cached tokens. But Assignment 1 does not require KV cache.

---

## 7. Training System

The training script performs the following steps:

```text
load config / arguments
    ↓
load tokenized train.dat and valid.dat with np.memmap
    ↓
initialize model
    ↓
initialize AdamW optimizer
    ↓
optionally load checkpoint
    ↓
for each train step:
        sample batch
        forward pass
        compute cross entropy loss
        backward pass
        gradient clipping
        optimizer step
        logging
        validation
        checkpoint saving
```

Training is controlled by `train_steps`, not epochs.

Approximate training budget:

```text
tokens_processed = train_steps × batch_size × context_length
```

This is more natural for language model pretraining because the data is treated as a long token stream.

Important training arguments:

| Argument | Meaning |
|---|---|
| `vocab_size` | Vocabulary size of the tokenizer and model |
| `context_len` | Sequence length used during training |
| `batch_size` | Number of sequences per step |
| `train_steps` | Number of optimizer updates |
| `max_lr` | Maximum learning rate |
| `min_lr` | Minimum learning rate |
| `warm_up_it` | Number of warmup steps |
| `cosine_it` | Step where cosine decay reaches `min_lr` |
| `val_interval` | Validation frequency |
| `save_intervals` | Checkpoint saving frequency |

Loss and perplexity:

```text
PPL = exp(loss)
```

So if:

```text
loss ≈ 2.3
```

then:

```text
PPL ≈ 10
```

---

## 8. Generation Flow

Generation uses the trained tokenizer and trained model checkpoint.

Generation is different from training.

Training:

```text
input sequence
    ↓
model forward
    ↓
logits for all positions
    ↓
loss against known targets
```

Generation:

```text
prompt
    ↓
encode prompt into token ids
    ↓
model forward
    ↓
take last-position logits
    ↓
sample one token
    ↓
append token
    ↓
repeat
```

A single model forward pass does not increase sequence length.

The sequence grows only inside the generation loop.

Pseudo-code:

```python
ids = tokenizer.encode(prompt)

for step in range(max_new_tokens):
    input_ids = ids[-context_length:]
    logits = model(input_ids)
    next_logits = logits[-1]
    next_id = sample(next_logits)
    ids.append(next_id)

text = tokenizer.decode(ids)
```

Supported decoding controls:

| Parameter | Meaning |
|---|---|
| `max_new_tokens` | Maximum number of generated tokens |
| `temperature` | Controls randomness of sampling |
| `top_p` | Keeps the smallest set of high-probability tokens whose probability mass exceeds `p` |
| `<|endoftext|>` | Optional stop token |

---

## 9. Current Results

### 9.1 Tokenizer Results

| Item | Value |
|---|---:|
| Dataset | TinyStories |
| Vocab size | 10000 |
| Special token | `<|endoftext|>` |
| Train tokens | TODO |
| Valid tokens | TODO |
| Longest token | TODO |
| Tokenizer training time | TODO |
| Dataset tokenization time | TODO |

---

### 9.2 Model Configuration

| Item | Value |
|---|---:|
| Dataset | TinyStories |
| Vocab size | 10000 |
| Context length | 256 |
| d_model | 512 |
| d_ff | 1344 |
| num_layers | 4 |
| num_heads | 16 |
| batch_size | TODO |
| train_steps | TODO |
| tokens processed | TODO |
| best validation loss | TODO |
| best validation PPL | TODO |

---

### 9.3 Training Curve

TODO: add training and validation loss curve.

Planned path:

```text
results/assignment1/loss_curves/tinystories_baseline.png
```

---

### 9.4 Generation Samples

TODO: add generated text samples.

Template:

```text
Prompt:
Once upon a time

Generated:
TODO
```

---

## 10. Experiments

Planned and completed experiments:

| Experiment | Change | Status | Result |
|---|---|---|---|
| Baseline | Default TinyStories config | TODO | TODO |
| Learning-rate sweep | Vary `max_lr` | TODO | TODO |
| Batch-size sweep | Vary `batch_size` | TODO | TODO |
| RoPE vs NoPE | Remove RoPE | TODO | TODO |
| RMSNorm ablation | Remove RMSNorm | TODO | TODO |
| Pre-norm vs post-norm | Change normalization placement | TODO | TODO |
| SwiGLU vs SiLU | Replace gated FFN | TODO | TODO |
| OpenWebText run | Train on OWT | TODO | TODO |

Experiment logging template:

```text
Experiment:
Purpose:
Change:
Command:
Result:
Observation:
Possible explanation:
```

---

## 11. Debugging Notes

### 11.1 Token id out of range

Symptom:

```text
IndexError: index out of range in self
```

Likely cause:

```text
max token id >= model vocab_size
```

Check:

```python
import numpy as np

x = np.memmap("train.dat", dtype=np.uint16, mode="r")
print(x.max())
```

The tokenizer vocabulary size, tokenized dataset, and model `vocab_size` must be consistent.

---

### 11.2 NumPy / Tensor mismatch in validation

Symptom:

```text
TypeError: expected np.ndarray (got Tensor)
```

Likely cause:

```text
data_loading already returns torch.Tensor,
but validation code still calls torch.from_numpy(...)
```

Fix:

```python
val_input_ids, val_target_ids = data_loading(
    val_data,
    batch_size,
    context_length,
    device=device,
)

val_input_ids = val_input_ids.long()
val_target_ids = val_target_ids.long()
```

---

### 11.3 RoPE position handling

RoPE needs token positions, but positions should not be stored in the dataset.

For standard training:

```python
B, S = input_ids.shape
positions = torch.arange(S, device=input_ids.device)
positions = positions.unsqueeze(0).expand(B, S)
```

For future KV-cache inference, positions should start from `past_seq_len`.

---

### 11.4 Training vs generation

Training does not call generation.

Training uses:

```text
input ids → logits for every position → cross entropy loss
```

Generation uses:

```text
prompt → logits at last position → sample one token → append → repeat
```

---

## 12. Lessons Learned

Main lessons from this milestone:

1. A language model pipeline starts before the model: tokenizer training and dataset tokenization are essential.
2. The tokenized dataset stores only token ids, not token strings or token positions.
3. Decoder-only LM training uses teacher forcing and predicts all next-token distributions in parallel.
4. Autoregressive generation samples one token at a time.
5. A single Transformer forward pass preserves sequence length.
6. Sequence length increases only in the generation loop.
7. RoPE positions belong to the model forward pass, not the tokenizer.
8. `train_steps × batch_size × context_length` is the useful training-budget estimate for language models.
9. `np.memmap` is useful for large tokenized datasets.
10. Many bugs come from shape mismatch, dtype mismatch, or inconsistent vocabulary sizes.
11. Loss and perplexity are directly related: `PPL = exp(loss)`.
12. Single-batch overfitting is a useful sanity check for Transformer implementation correctness.

---

## 13. Next Steps

Immediate next steps:

- Finish baseline TinyStories training.
- Add final validation loss and perplexity.
- Add training and validation loss curves.
- Add generated text samples.
- Run learning-rate sweep.
- Run batch-size sweep.
- Run ablation experiments:
  - RoPE vs NoPE
  - RMSNorm vs no RMSNorm
  - pre-norm vs post-norm
  - SwiGLU vs SiLU
- Extend experiments to OpenWebText.
- Prepare a concise resume version of this project.

---

## 14. Resume Summary Draft

Possible resume description:

```text
Implemented a GPT-style language model training pipeline from scratch, including byte-level BPE tokenizer training, token-id memmap dataset construction, decoder-only Transformer LM with RMSNorm/RoPE/causal attention/SwiGLU, AdamW optimization, checkpointing, validation, and autoregressive text generation. Trained and evaluated small language models on TinyStories/OpenWebText and documented system-level debugging and ablation experiments.
```