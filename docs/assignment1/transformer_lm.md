# Transformer Language Model Architecture

This note explains the decoder-only Transformer language model used in
Assignment 1.

The goal is to understand how token ids become logits over the vocabulary, and
how each module changes the tensor flowing through the model.

In this project, the architecture is a GPT-style decoder-only Transformer:

```text
token ids
  ↓
token embedding
  ↓
Transformer block × num_layers
  ↓
final RMSNorm
  ↓
linear LM head
  ↓
logits over vocabulary
```

The implementation lives mainly in:

| File | Purpose |
|---|---|
| `cs336_basics/model/modules.py` | Core neural network layers and attention primitives |
| `cs336_basics/model/transformer.py` | Transformer block and full language model |
| `cs336_basics/train.py` | Initializes the model and trains it with next-token prediction |
| `cs336_basics/generate.py` | Loads a trained model and generates text autoregressively |
| `tests/test_model.py` | Snapshot tests for individual modules and the full Transformer LM |
| `tests/adapters.py` | Adapter functions that connect tests to implementation modules |

Document map:

| Section | Core content |
|---|---|
| [1. What the Language Model Computes](#1-what-the-language-model-computes) | The model interface: token ids in, vocabulary logits out |
| [2. Architecture Overview](#2-architecture-overview) | High-level decoder-only Transformer data flow and tensor shapes |
| [3. Function and Module Map](#3-function-and-module-map) | Table of implementation classes/functions and their data operations |
| [4. Token Embedding and LM Head](#4-token-embedding-and-lm-head) | How token ids become vectors and vectors become logits |
| [5. Transformer Block](#5-transformer-block) | Pre-norm residual block: RMSNorm, attention, SwiGLU, residual connections |
| [6. Causal Multi-head Self-attention](#6-causal-multi-head-self-attention) | Q/K/V projection, head splitting, attention data flow, causal mask, softmax, training vs generation behavior |
| [7. RoPE Positional Information](#7-rope-positional-information) | How `token_positions` and pairwise vector rotation inject position into Q/K vectors |
| [8. Feed-forward Network: SwiGLU](#8-feed-forward-network-swiglu) | The gated MLP used after attention in each block |
| [9. Full Forward Pass](#9-full-forward-pass) | End-to-end shape trace through `transformer_lm.forward` |
| [10. Training Interface](#10-training-interface) | How logits are used for next-token prediction loss |
| [11. Generation Interface](#11-generation-interface) | How the same model is reused one step at a time for sampling |
| [12. Important Invariants](#12-important-invariants) | Shape, masking, vocabulary, and position assumptions that must hold |
| [13. Summary](#13-summary) | Compact recap of the model architecture |

---

## 1. What the Language Model Computes

The Transformer language model receives token ids:

```text
input_ids: (B, S)
```

where:

```text
B = batch size
S = sequence length / context length
```

It returns logits:

```text
logits: (B, S, vocab_size)
```

Each position produces a score for every token in the vocabulary.

For a sequence:

```text
tokens:     [x0, x1, x2, x3, x4]
input_ids:  [x0, x1, x2, x3]
targets:    [x1, x2, x3, x4]
```

the model learns:

```text
position 0: predict x1 from x0
position 1: predict x2 from x0, x1
position 2: predict x3 from x0, x1, x2
position 3: predict x4 from x0, x1, x2, x3
```

This is why the model is called a language model: it estimates a next-token
distribution for each position.

---

## 2. Architecture Overview

The model is decoder-only. It does not have an encoder and it does not use
cross-attention.

The high-level computation is:

```text
input_ids: (B, S)
  ↓
Embedding
  ↓
x: (B, S, d_model)
  ↓
TransformerBlock × N
  ↓
x: (B, S, d_model)
  ↓
RMSNorm
  ↓
x: (B, S, d_model)
  ↓
Linear LM head
  ↓
logits: (B, S, vocab_size)
```

Each Transformer block uses pre-normalization:

```text
x
  ↓
x + Attention(RMSNorm(x))
  ↓
x + SwiGLU(RMSNorm(x))
  ↓
block output
```

The attention is causal, so position `i` can attend only to positions
`0 ... i`.

Important dimensions:

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `S` | sequence length |
| `vocab_size` | number of tokenizer vocabulary entries |
| `d_model` | hidden-state width |
| `num_heads` | number of attention heads |
| `d_k` | per-head attention dimension, `d_model // num_heads` |
| `d_ff` | hidden dimension inside the SwiGLU feed-forward network |
| `num_layers` | number of Transformer blocks |

---

## 3. Function and Module Map

The model is built from small modules in `modules.py` and assembled in
`transformer.py`.

| Function / module | Input | Operation on data | Output |
|---|---|---|---|
| `Embedding.forward(token_ids)` | token ids with shape `(...,)` | Converts ids to integer dtype if needed and indexes embedding table | `(..., d_model)` vectors |
| `Linear.forward(x)` | tensor with final dim `in_dim` | Applies learned matrix multiplication with weight `(out_dim, in_dim)` | tensor with final dim `out_dim` |
| `RMSNorm.forward(x)` | hidden states `(..., d_model)` | Normalizes by root-mean-square over final dim and scales by learned weight | same shape as input |
| `RotaryPositionalEmbedding.forward(x, token_positions)` | Q/K tensor and optional positions | Computes cos/sin rotations and rotates pairs of hidden dims | same shape as input |
| `softmax(x, dim)` | unnormalized scores | Subtracts max for numerical stability, exponentiates, and normalizes along `dim` | probabilities that sum to 1 along `dim` |
| `scaled_dot_product_attention(q, k, v, mask)` | Q/K/V tensors and optional mask | Computes attention scores, applies mask, softmaxes over key positions, and takes weighted value sum | attention output |
| `multihead_self_attention.forward(x, token_positions)` | hidden states `(B, S, d_model)` | Projects Q/K/V, splits heads, applies RoPE, applies causal attention, merges heads | `(B, S, d_model)` |
| `SwiGLU.forward(x)` | hidden states `(..., d_model)` | Applies gated feed-forward transformation | `(..., d_model)` |
| `transformer_block.forward(in_features, token_positions)` | block input `(B, S, d_model)` | Applies pre-norm attention residual, then pre-norm FFN residual | `(B, S, d_model)` |
| `transformer_lm.forward(x, token_positions)` | token ids `(B, S)` | Embeds ids, applies all blocks, normalizes, projects to vocab logits | `(B, S, vocab_size)` |

The full model stores these main submodules:

| Attribute | Module | Role |
|---|---|---|
| `token_embedding` | `Embedding` | Token id to hidden vector |
| `layers` | `nn.ModuleList[transformer_block]` | Stack of decoder blocks |
| `output_norm` | `RMSNorm` | Final hidden-state normalization |
| `output_embedding` | `Linear` | LM head from hidden state to vocabulary logits |

---

## 4. Token Embedding and LM Head

### 4.1 Token Embedding

The tokenizer produces integer token ids. The Transformer cannot operate
directly on ids, so the first step is an embedding lookup.

In `Embedding`:

```text
weight: (vocab_size, d_model)
token_ids: (B, S)
output: (B, S, d_model)
```

For each token id, the model selects one row from the embedding table.

Example:

```text
input_ids[b, s] = 42
  ↓
embedding vector = weight[42]
```

This creates the initial hidden states for the Transformer blocks.

---

### 4.2 LM Head

At the end of the model, hidden states are projected back to vocabulary-sized
logits:

```text
hidden: (B, S, d_model)
  ↓
Linear(d_model, vocab_size)
  ↓
logits: (B, S, vocab_size)
```

Each vector `logits[b, s, :]` contains one score per possible next token.

The model does not apply softmax inside `transformer_lm.forward`. Training uses
cross entropy directly on logits, and generation applies softmax after
temperature or top-p processing.

---

## 5. Transformer Block

Each block in `transformer.py` has:

```text
ln1
attn
ln2
ffn
```

The block forward pass is:

```python
x = in_features
x = x + self.attn(self.ln1(x), token_positions=token_positions)
x = x + self.ffn(self.ln2(x))
return x
```

As a data flow:

```text
x: (B, S, d_model)
  ↓
RMSNorm
  ↓
causal multi-head self-attention with RoPE
  ↓
add residual connection
  ↓
RMSNorm
  ↓
SwiGLU feed-forward network
  ↓
add residual connection
  ↓
block output: (B, S, d_model)
```

The residual connections preserve the main information stream and allow each
submodule to add an update rather than replacing the whole representation.

The block is pre-norm because normalization happens before attention and before
the feed-forward network.

---

## 6. Causal Multi-head Self-attention

Self-attention lets each token mix information from earlier tokens in the same
sequence.

In a decoder-only language model, attention must be causal:

```text
token at position i may attend to positions <= i
token at position i must not attend to positions > i
```

The complete attention data flow is:

```text
x: (B, S, d_model)
  ↓
linear projections
  ├── q = Wq(x): (B, S, d_model)
  ├── k = Wk(x): (B, S, d_model)
  └── v = Wv(x): (B, S, d_model)
  ↓
split into heads
  ├── q: (B, H, S, d_k)
  ├── k: (B, H, S, d_k)
  └── v: (B, H, S, d_k)
  ↓
apply RoPE to q and k
  ├── q: (B, H, S, d_k)
  ├── k: (B, H, S, d_k)
  └── v: unchanged
  ↓
attention scores
  scores = q @ k^T / sqrt(d_k): (B, H, S, S)
  ↓
causal mask
  future positions become -inf
  ↓
softmax over key positions
  attention probabilities: (B, H, S, S)
  ↓
weighted sum of values
  attention output: (B, H, S, d_k)
  ↓
merge heads
  (B, S, H * d_k) = (B, S, d_model)
  ↓
output projection
  (B, S, d_model)
```

---

### 6.1 Q/K/V Projection

The attention module starts from:

```text
x: (B, S, d_model)
```

It applies three learned linear projections:

```text
q = Wq x
k = Wk x
v = Wv x
```

Each has shape:

```text
(B, S, d_model)
```

Then the final dimension is split into heads:

```text
(B, S, d_model)
  ↓
(B, num_heads, S, d_k)
```

where:

```text
d_k = d_model // num_heads
```

The assertion:

```python
assert d_model % num_heads == 0
```

ensures the head dimension is an integer.

---

### 6.2 Causal Mask

The causal mask is lower triangular:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

The implementation creates it as:

```python
torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
```

and reshapes it to:

```text
(1, 1, S, S)
```

This broadcasts across batch and heads.

Masked positions are filled with `-inf` before softmax:

```python
q_k_score = q_k_score.masked_fill(mask == False, float("-inf"))
```

After softmax, future positions receive zero probability.

Example for sequence length 4:

| Query position | Visible key positions | Mask row |
|---|---|---|
| `0` | `0` | `1 0 0 0` |
| `1` | `0, 1` | `1 1 0 0` |
| `2` | `0, 1, 2` | `1 1 1 0` |
| `3` | `0, 1, 2, 3` | `1 1 1 1` |

The row is the query position. The column is the key/value position. A `1`
means the query may attend to that key. A `0` means that key is in the future
and must be hidden.

---

### 6.3 Causal Mask During Training and Generation

The purpose of the mask is the same in training and generation:

```text
do not let a token see future tokens
```

But the way we use the model is different.

During training, the model receives a whole sequence:

```text
input_ids: (B, S)
```

and returns logits for every position:

```text
logits: (B, S, vocab_size)
```

The causal mask is essential because all positions are computed in parallel.
Without the mask, position `0` could directly attend to `x1`, position `1`
could directly attend to `x2`, and the next-token task would leak the answer.

Training mask shape in this implementation:

```text
mask: (1, 1, S, S)
scores: (B, H, S, S)
```

During generation in this repository, `generate.py` does not use a KV cache.
At every generation step it takes the current context window and recomputes the
full forward pass:

```text
context_tokens = generated_tokens[-model.context_length:]
input_ids: (1, S_current)
logits: (1, S_current, vocab_size)
use logits[0, -1, :]
```

So the mask construction is still lower triangular over the current context:

```text
mask: (1, 1, S_current, S_current)
```

The difference is semantic: generation only uses the final position's logits.
Earlier positions are recomputed as context, but their logits are discarded.

If we implemented KV-cache generation, the mask would be conceptually
different. The model would usually process only the newest query token:

```text
q: (B, H, 1, d_k)
k/v cache: (B, H, S_past + 1, d_k)
scores: (B, H, 1, S_past + 1)
```

In that case, the single new query should attend to all cached past keys and
itself. There are no future keys in the cache, so the mask is often all visible
for that one query, or it is built as a rectangular causal mask aligned to the
absolute positions.

---

### 6.4 Scaled Dot-product Attention Scores

The attention scores are:

```text
scores = Q K^T / sqrt(d_k)
```

Shape trace:

```text
q:      (B, H, S, d_k)
k:      (B, H, S, d_k)
scores: (B, H, S, S)
```

Each row `scores[b, h, query_position, :]` contains the compatibility between
one query token and all key tokens.

These values are still logits, not probabilities. They can be positive,
negative, large, small, and they do not sum to 1.

---

### 6.5 Softmax over Key Positions

After masking, attention uses softmax over the key dimension:

```text
attention_probabilities = softmax(scores, dim=-1)
```

In `modules.py`, softmax is implemented as:

```python
def softmax(x, dim):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)
```

The subtraction:

```text
x = x - max(x)
```

does not change the final probability distribution, but it makes the exponent
calculation numerically safer by preventing very large values from overflowing.

Softmax turns scores into a probability distribution:

```text
scores:      [2.0, 1.0, -inf, -inf]
softmax:     [0.731, 0.269, 0.000, 0.000]
sum:          1.000
```

The mask must be applied before softmax:

```text
future score → -inf → exp(-inf) = 0
```

If we applied softmax first and then multiplied masked positions by zero, the
remaining probabilities would no longer sum to 1 unless we normalized again.
That would be the wrong attention distribution.

Shape:

```text
attention probabilities: (B, H, S, S)
```

For every query position, the probabilities over key positions sum to 1:

```text
sum over final dimension = 1
```

---

### 6.6 Weighted Sum of Values

The attention probabilities weight the values:

```text
v:      (B, H, S, d_k)
output: (B, H, S, d_k)
```

Finally, heads are merged:

```text
(B, H, S, d_k)
  ↓
(B, S, H * d_k)
  ↓
(B, S, d_model)
```

and an output projection `w_o` mixes information across heads.

---

## 7. RoPE Positional Information

Self-attention by itself is permutation-insensitive. The model needs position
information to distinguish:

```text
"dog bites man"
"man bites dog"
```

This implementation uses RoPE, or Rotary Positional Embedding.

RoPE is applied to Q and K, not V:

```text
q_i = RoPE(q_i, token_positions)
k_i = RoPE(k_i, token_positions)
v_i = unchanged
```

The tokenizer does not create token positions. It only produces token ids. The
position information belongs to the model path:

```text
tokenizer:
    text → token ids

model:
    token ids + token positions → logits
```

The model receives optional `token_positions`, and if they are not provided,
RoPE uses:

```text
[0, 1, 2, ..., S - 1]
```

In the current code, `token_positions` is passed through:

```text
transformer_lm.forward(x, token_positions)
  ↓
transformer_block.forward(..., token_positions)
  ↓
multihead_self_attention.forward(..., token_positions)
  ↓
RotaryPositionalEmbedding.forward(q_or_k, token_positions)
```

---

### 7.1 What `token_positions` Means

`token_positions` tells RoPE the logical position of each token in the sequence.

For ordinary training, each sampled sequence is treated as a fresh context:

```text
input_ids:       [x0, x1, x2, x3]
token_positions: [0,  1,  2,  3]
```

So `token_positions=None` is enough, because the RoPE module creates:

```python
torch.arange(S, device=x.device)
```

For current generation in `generate.py`, the model also recomputes a full
context window each step:

```text
context_tokens:  [y0, y1, y2, ..., yS-1]
token_positions: [0,  1,  2,  ..., S-1]
```

Again, `token_positions=None` is enough for this implementation.

The cases where explicit `token_positions` become important are:

| Case | Why default `[0, ..., S-1]` is not enough |
|---|---|
| KV-cache generation | The newest token may logically be at absolute position `S_past`, even though the query length is only `1` |
| Processing a slice of a longer sequence | A slice may start at position `1000`, not `0` |
| Different batch samples have different offsets | Each sample may need its own position row |

Example with KV cache:

```text
cached context length = 5
new token query shape = (B, H, 1, d_k)
correct token_positions = [5]
wrong default position = [0]
```

If we used position `0` for the new token, RoPE would rotate Q/K as if this
token were at the beginning of the sequence, which breaks the positional
meaning of attention.

---

### 7.2 RoPE Frequency and Angle

RoPE rotates each pair of dimensions by a position-dependent angle.

In this implementation, the model first splits `d_model` into attention heads:

```text
d_model = num_heads × d_k
```

RoPE is then applied inside each head, along the final `d_k` dimension of Q and
K:

```text
q/k: (B, H, S, d_k)
```

So the "dimension position" used by RoPE is the index inside each head's
`d_k`, not the original unsplit `d_model` vector as one giant block.

For a head dimension:

```text
d_k = 8
```

the dimensions are grouped as:

```text
(0, 1), (2, 3), (4, 5), (6, 7)
```

Each pair has its own inverse frequency:

```text
inv_freq[j] = 1 / theta^(2j / d_k)
```

In the code:

```python
self.inv_freq = 1 / (theta ** (torch.arange(0, d_k, 2) / d_k))
```

For token position `p`, the angle for pair `j` is:

```text
angle[p, j] = p * inv_freq[j]
```

So position and hidden dimension both matter:

```text
same dimension pair + different token position → different angle
same token position + different dimension pair → different angle
```

This is why RoPE is not just "add position somewhere." It rotates each Q/K
vector differently depending on both:

1. where the token is in the sequence;
2. which pair of dimensions is being rotated.

---

### 7.3 Two-dimensional Rotation Formula

For one pair of hidden dimensions:

```text
x_pair = [x_0, x_1]
```

and angle `α`, a standard 2D rotation is:

```text
[x_0']   [ cos α  -sin α ] [x_0]
[x_1'] = [ sin α   cos α ] [x_1]
```

which means:

```text
x_0' = x_0 cos α - x_1 sin α
x_1' = x_0 sin α + x_1 cos α
```

The implementation writes the same operation in vector form.

First, `rotate_tensor` converts:

```text
(x_even, x_odd) → (-x_odd, x_even)
```

Then RoPE computes:

```text
output = x * cos + rotate_tensor(x) * sin
```

For one pair:

```text
x * cos:
    (x_0 cos α, x_1 cos α)

rotate_tensor(x) * sin:
    (-x_1 sin α, x_0 sin α)

sum:
    (x_0 cos α - x_1 sin α,
     x_1 cos α + x_0 sin α)
```

This matches the 2D rotation matrix.

---

### 7.4 Shape Broadcasting in the Implementation

In attention, RoPE receives Q or K shaped:

```text
x: (B, H, S, d_k)
```

If `token_positions` is omitted:

```text
token_positions: (S,)
```

The code computes:

```text
theta = token_positions × inv_freq
```

with shape:

```text
theta: (S, d_k / 2)
```

Then:

```text
cos: (S, d_k)
sin: (S, d_k)
```

because each angle is repeated for the two dimensions in its pair:

```python
cos = theta.cos().repeat_interleave(2, dim=-1)
sin = theta.sin().repeat_interleave(2, dim=-1)
```

The code then unsqueezes until `cos` and `sin` can broadcast over `x`:

```text
cos/sin: (1, 1, S, d_k)
x:       (B, H, S, d_k)
```

This is why the same position angles are shared across batch and heads by
default.

If `token_positions` has shape `(B, S)`, then the angles can differ per batch
sample:

```text
token_positions: (B, S)
theta:           (B, S, d_k / 2)
cos/sin:         (B, 1, S, d_k)
x:               (B, H, S, d_k)
```

---

### 7.5 Training and Generation Positions

For normal Assignment 1 training without KV cache, positions usually start at
0 for every sequence in the batch.

```text
training input:   (B, S)
token positions:  [0, 1, 2, ..., S-1]
```

For current generation in this repository, each step rebuilds the context
window and calls the model on that whole window:

```text
generation context: [last S_current generated tokens]
token positions:    [0, 1, 2, ..., S_current - 1]
```

This means the current implementation uses positions relative to the current
window. That is consistent with how the forward pass is called here.

For KV-cache generation, the model would not recompute the entire window. It
would pass only the newest token but still need its absolute logical position:

```text
past tokens:       positions [0, 1, 2, 3, 4]
new token only:    position  [5]
```

That is the main reason `token_positions` exists as an optional argument even
though ordinary training can omit it.

---

## 8. Feed-forward Network: SwiGLU

After attention, each position is processed independently by the feed-forward
network.

This implementation uses SwiGLU instead of a plain two-layer MLP.

The formula in `SwiGLU.forward` is:

```text
output = W2( SiLU(W1 x) * (W3 x) )
```

Shape trace:

```text
x:          (..., d_model)
W1 x:       (..., d_ff)
SiLU(W1 x): (..., d_ff)
W3 x:       (..., d_ff)
gate:       (..., d_ff)
W2 gate:    (..., d_model)
```

The multiplication is elementwise. One branch creates an activated hidden
representation, and the other branch gates it.

In the Transformer block, the SwiGLU output is added back to the residual
stream:

```text
x = x + SwiGLU(RMSNorm(x))
```

---

## 9. Full Forward Pass

The full model is implemented in `transformer_lm.forward`.

Input:

```text
x: (B, S)
```

where `x` contains token ids.

Step-by-step:

```text
x = token_embedding(x)
```

Shape:

```text
(B, S) → (B, S, d_model)
```

Then each block updates the hidden states:

```text
for layer in layers:
    x = layer(x, token_positions=token_positions)
```

Shape stays:

```text
(B, S, d_model)
```

Then final normalization and projection:

```text
x = output_norm(x)
x = output_embedding(x)
```

Shape:

```text
(B, S, d_model) → (B, S, vocab_size)
```

The returned tensor is logits, not probabilities.

---

## 10. Training Interface

Training uses the full output sequence in parallel.

The data loader samples:

```text
input_ids:  (B, S)
target_ids: (B, S)
```

where targets are shifted by one token:

```text
input_ids:  [x_i,   x_i+1, ..., x_i+S-1]
target_ids: [x_i+1, x_i+2, ..., x_i+S]
```

The model computes:

```text
logits = model(input_ids)
```

with shape:

```text
(B, S, vocab_size)
```

Cross entropy compares each position's logits against the corresponding target
token.

The causal mask is what makes this valid: even though all positions are
computed in one forward pass, position `s` cannot see target token `s + 1` or
anything after it.

During this training setup, `token_positions` is usually omitted. RoPE then
uses positions `[0, 1, ..., S - 1]` for every sampled sequence. This matches the
fact that each sampled sequence is trained as an independent context window.

---

## 11. Generation Interface

Generation uses the same model but runs it autoregressively.

The loop in `generate.py` does:

```text
prompt text
  ↓
tokenizer.encode(prompt)
  ↓
generated_tokens
  ↓
take last context_length tokens
  ↓
model(input_ids)
  ↓
take logits at final position
  ↓
temperature / top-p sampling
  ↓
append sampled token
  ↓
repeat
```

At each step:

```python
context_tokens = generated_tokens[-model.context_length:]
input_ids = torch.tensor(context_tokens).unsqueeze(0)
logits = model(input_ids)
next_token_logits = logits[0, -1, :]
```

Only the final position's logits are used because generation needs only the
next token distribution after the current context.

Unlike training, generation does not compute a loss. It samples token ids and
then uses the tokenizer to decode them back into text.

In the current implementation, generation recomputes the whole context window
at every step, so the default RoPE positions and lower-triangular causal mask
are still valid for that window. If generation were changed to use a KV cache,
the newest token would need explicit `token_positions`, and the attention mask
would be built for one query attending over cached keys.

---

## 12. Important Invariants

The Transformer LM relies on these invariants:

1. `input_ids` contains token ids in `[0, vocab_size)`.
2. The tokenizer vocabulary size and model `vocab_size` match.
3. `d_model % num_heads == 0`, so each head has integer dimension `d_k`.
4. The attention mask is causal, so tokens cannot attend to future tokens.
5. RoPE is applied to Q and K, not V.
6. Transformer blocks preserve shape `(B, S, d_model)`.
7. The final model output has shape `(B, S, vocab_size)`.
8. The forward pass returns logits, not probabilities.
9. Token positions belong to the model/RoPE path, not the tokenizer or `.dat`
   dataset files.

---

## 13. Summary

The Transformer language model is the neural part of the Assignment 1 pipeline.

The tokenizer turns text into token ids. The model turns token ids into
next-token logits.

The key ideas are:

- Token ids are converted to dense vectors by an embedding table.
- A stack of pre-norm Transformer blocks updates the hidden states.
- Each block contains causal multi-head self-attention with RoPE and a SwiGLU
  feed-forward network.
- Residual connections preserve and update the hidden-state stream.
- The final RMSNorm and linear head project hidden states to vocabulary logits.
- Training uses all positions in parallel with shifted targets.
- Generation reuses the same model one token at a time by sampling from the
  final-position logits.
