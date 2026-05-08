# Text Generation

This note explains the generation path for the Assignment 1 Transformer
language model.

The goal is to understand how a trained checkpoint and tokenizer are used to
turn a prompt into generated text.

Generation is different from training:

```text
training:
    known input/target batches
    parallel next-token prediction
    loss + backprop + optimizer step

generation:
    prompt text
    sample one token at a time
    append sampled token to context
    no loss, no gradients
```

The high-level generation pipeline is:

```text
prompt text
  ↓
tokenizer.encode(prompt)
  ↓
prompt token ids
  ↓
trained Transformer LM
  ↓
logits at final position
  ↓
temperature softmax
  ↓
top-p filtering
  ↓
sample next token id
  ↓
append token id
  ↓
repeat
  ↓
tokenizer.decode(generated ids)
  ↓
generated text
```

The implementation lives mainly in:

| File | Purpose |
|---|---|
| `cs336_basics/generate.py` | Main text generation script |
| `cs336_basics/model/transformer.py` | Transformer LM architecture used for inference |
| `cs336_basics/tokenizer.py` | Encode prompt text and decode generated token ids |
| `cs336_basics/check_pointing.py` | Training checkpoint format used by generation |
| `docs/assignment1/BPE-tokenizer.md` | Tokenizer training/encode/decode explanation |
| `docs/assignment1/transformer_lm.md` | Transformer model architecture explanation |

Document map:

| Section | Core content |
|---|---|
| [1. What Generation Does](#1-what-generation-does) | Autoregressive next-token sampling from a trained model |
| [2. Generation Data Flow](#2-generation-data-flow) | End-to-end flow from prompt string to generated text |
| [3. Function and Module Map](#3-function-and-module-map) | Table of generation functions and data operations |
| [4. Loading Model and Tokenizer](#4-loading-model-and-tokenizer) | Checkpoint loading, tokenizer loading, model reconstruction |
| [5. Prompt Encoding and Context Window](#5-prompt-encoding-and-context-window) | Prompt tokenization and `context_length` truncation |
| [6. Model Forward Pass During Generation](#6-model-forward-pass-during-generation) | Why generation uses only final-position logits |
| [7. Temperature Scaling](#7-temperature-scaling) | How temperature changes the probability distribution |
| [8. Top-p Sampling](#8-top-p-sampling) | Nucleus sampling and probability renormalization |
| [9. Autoregressive Loop and Stop Conditions](#9-autoregressive-loop-and-stop-conditions) | Appending sampled tokens and stopping on max tokens or EOS |
| [10. Training vs Generation](#10-training-vs-generation) | Differences in gradients, targets, logits usage, and context |
| [11. Important Invariants](#11-important-invariants) | Checkpoint/tokenizer/model/sampling assumptions |
| [12. Example Command](#12-example-command) | Example CLI usage |
| [13. Summary](#13-summary) | Compact recap of generation |

---

## 1. What Generation Does

Generation uses a trained language model to sample text one token at a time.

The model receives a context:

```text
[x0, x1, x2, ..., xt]
```

and returns logits for every position:

```text
logits: (1, S, vocab_size)
```

For generation, only the final position matters:

```text
next_token_logits = logits[0, -1, :]
```

Those logits represent the model's prediction for:

```text
P(next token | current context)
```

After sampling one token, generation appends it to the context and repeats.

This is called autoregressive generation:

```text
prompt
  ↓
predict token 1
  ↓
prompt + token 1
  ↓
predict token 2
  ↓
prompt + token 1 + token 2
  ↓
...
```

---

## 2. Generation Data Flow

The core loop in `generate_text` is:

```text
prompt: str
  ↓
tokenizer.encode(prompt)
  ↓
prompt_tokens: list[int]
  ↓
generated_tokens = prompt_tokens.copy()
  ↓
for each generation step:
    context_tokens = generated_tokens[-model.context_length:]
    input_ids = tensor(context_tokens).unsqueeze(0)
    logits = model(input_ids)
    next_token_logits = logits[0, -1, :]
    probabilities = softmax_with_temperature(next_token_logits, temperature)
    probabilities = top_p_sampling(probabilities, top_p)
    next_token = torch.multinomial(probabilities, 1)
    generated_tokens.append(next_token)
    optionally stop on eos_token
  ↓
tokenizer.decode(generated_tokens)
  ↓
generated text
```

Important shape trace:

```text
context_tokens:    list[int] length S_current
input_ids:         (1, S_current)
logits:            (1, S_current, vocab_size)
next_token_logits: (vocab_size)
probabilities:     (vocab_size)
next_token:        scalar int
```

This implementation does not use KV cache. It recomputes the full current
context window on every generation step.

---

## 3. Function and Module Map

| Function / object | Input | Operation on data/state | Output |
|---|---|---|---|
| `main()` | CLI arguments | Parses paths, model args, sampling args, device, and special tokens | prints generated samples |
| `load_model_and_tokenizer(args, device)` | checkpoint path, vocab path, merges path, config args | Loads tokenizer, checkpoint, model config, model weights | `(model, tokenizer)` |
| `Tokenizer.from_files(...)` | `vocab.pkl`, `merges.pkl`, special tokens | Reconstructs runtime tokenizer | tokenizer |
| `torch.load(args.checkpoint, map_location=device)` | checkpoint file | Loads saved model/optimizer/config dictionary or raw state dict | checkpoint object |
| `build_model_config(args, checkpoint)` | CLI args and checkpoint | Combines defaults, checkpoint config, and CLI model args | model config dict |
| `transformer_lm(...)` | model config | Rebuilds model architecture compatible with checkpoint | model |
| `get_checkpoint_model_state(checkpoint)` | checkpoint object | Extracts `model_state` or `model_state_dict` if present | state dict |
| `model.load_state_dict(...)` | state dict | Loads trained parameters into model | initialized trained model |
| `generate_text(...)` | model, tokenizer, prompt, sampling config | Runs autoregressive sampling loop | generated text string |
| `tokenizer.encode(prompt)` | prompt string | Converts prompt text to token ids | `list[int]` |
| `model(input_ids)` | context token tensor `(1, S)` | Produces logits for every context position | `(1, S, vocab_size)` |
| `softmax_with_temperature(logits, temperature)` | final-position logits | Divides logits by temperature and applies softmax | probabilities |
| `top_p_sampling(probabilities, p)` | probability distribution | Keeps smallest high-probability nucleus whose cumulative mass reaches `p` | filtered probabilities |
| `torch.multinomial(probabilities, 1)` | probability distribution | Samples one token id | scalar token id |
| `tokenizer.decode(generated_tokens)` | generated token ids | Converts token ids back to text | generated string |

---

## 4. Loading Model and Tokenizer

Generation needs two trained artifacts:

```text
tokenizer artifacts:
    vocab.pkl
    merges.pkl

model artifact:
    checkpoint.pt
```

The tokenizer is loaded first:

```python
tokenizer = Tokenizer.from_files(
    vocab_filepath=args.vocab,
    merges_filepath=args.merges,
    special_tokens=args.special_tokens,
)
```

The checkpoint is loaded onto the selected device:

```python
checkpoint = torch.load(args.checkpoint, map_location=device)
```

The model config is built from:

1. `DEFAULT_MODEL_CONFIG`;
2. `checkpoint["model_config"]`, if the checkpoint contains it;
3. CLI/default model arguments.

Then the model is reconstructed:

```python
model = transformer_lm(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    num_layers=config["num_layers"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    rope_theta=config["rope_theta"],
    d_ff=config["d_ff"],
).to(device)
```

After construction, trained weights are loaded:

```python
model.load_state_dict(get_checkpoint_model_state(checkpoint))
model.eval()
```

`model.eval()` puts the model in inference mode. This is important for modules
such as dropout or batch norm in general. This model does not use dropout or
batch norm, but using eval mode is still the correct inference convention.

Important consistency requirement:

```text
checkpoint model config must match reconstructed model architecture
```

For example, a checkpoint trained with `d_model = 512` cannot be loaded into a
model built with `d_model = 768`.

---

## 5. Prompt Encoding and Context Window

Generation starts from a prompt string:

```python
prompt_tokens = tokenizer.encode(prompt)
```

If the prompt encodes to no tokens, generation stops with an error:

```python
if not prompt_tokens:
    raise ValueError("prompt must encode to at least one token")
```

The generated token list starts as a copy of the prompt:

```python
generated_tokens = prompt_tokens.copy()
```

At every generation step, the model sees only the last `context_length` tokens:

```python
context_tokens = generated_tokens[-model.context_length:]
```

This is necessary because the Transformer has a maximum context length.

If generated text grows longer than the context window, older tokens fall out
of the model input:

```text
generated_tokens:
    [old tokens ... recent tokens]

model input:
    [last context_length tokens only]
```

The list is converted into a batch of size 1:

```python
input_ids = torch.tensor(
    context_tokens,
    dtype=torch.long,
    device=device,
).unsqueeze(0)
```

Shape:

```text
context_tokens: list[int] length S_current
input_ids:      (1, S_current)
```

---

## 6. Model Forward Pass During Generation

The model computes logits for every input position:

```python
logits = model(input_ids)
```

Shape:

```text
logits: (1, S_current, vocab_size)
```

During generation, only the final position is used:

```python
next_token_logits = logits[0, -1, :]
```

Why final position?

The final position has attended to the whole current context. Its logits answer:

```text
What token should come after this full context?
```

Earlier positions answer older questions:

```text
position 0 predicts token 1
position 1 predicts token 2
...
final position predicts the next new token
```

Since we only need the next new token, all earlier logits are discarded.

This implementation recomputes the whole context every step. With KV cache,
the model would reuse previous K/V tensors and run only the newest token. That
optimization is not part of this Assignment 1 generation path.

---

## 7. Temperature Scaling

Raw logits are not probabilities. They are converted to probabilities with
softmax.

Before softmax, generation applies temperature:

```python
scaled_logits = logits / temperature
probabilities = torch.softmax(scaled_logits, dim=-1)
```

Temperature controls randomness.

| Temperature | Effect |
|---|---|
| `< 1` | Sharper distribution; more deterministic |
| `= 1` | Original model distribution |
| `> 1` | Flatter distribution; more random |

Example intuition:

```text
low temperature:
    likely tokens become much more dominant

high temperature:
    lower-probability tokens get more chance
```

The implementation rejects non-positive temperatures:

```python
if temperature <= 0:
    raise ValueError("temperature must be greater than 0")
```

---

## 8. Top-p Sampling

Top-p sampling is also called nucleus sampling.

Instead of sampling from the full vocabulary, it keeps the smallest set of
highest-probability tokens whose cumulative probability reaches `p`.

For example, if sorted probabilities are:

```text
token A: 0.40
token B: 0.30
token C: 0.15
token D: 0.10
token E: 0.05
```

and:

```text
top_p = 0.9
```

the nucleus is:

```text
A + B + C + D = 0.95
```

Token E is removed.

The implementation does:

```text
sort probabilities descending
  ↓
compute cumulative sum
  ↓
remove tokens after cumulative probability exceeds p
  ↓
keep at least the first token
  ↓
renormalize remaining probabilities
  ↓
scatter back to original vocabulary order
```

Key code:

```python
sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
remove_mask = cumulative_probs > p
remove_mask[..., 1:] = remove_mask[..., :-1].clone()
remove_mask[..., 0] = False
```

The shift:

```python
remove_mask[..., 1:] = remove_mask[..., :-1]
remove_mask[..., 0] = False
```

keeps the first token that pushes cumulative probability above `p`. This avoids
dropping too aggressively and guarantees at least one token remains.

After filtering:

```python
filtered_probs = filtered_probs / torch.sum(filtered_probs)
```

Renormalization is required because removed tokens have probability zero.

If:

```text
top_p = 1
```

the distribution is unchanged.

---

## 9. Autoregressive Loop and Stop Conditions

After temperature and top-p processing, generation samples one token:

```python
next_token = torch.multinomial(probabilities, num_samples=1).item()
generated_tokens.append(next_token)
```

Then it optionally checks for the end-of-sequence token:

```python
decoded_token = tokenizer.decode([next_token])
if decoded_token == eos_token:
    break
```

Generation stops when either:

1. `max_tokens` generated tokens have been sampled;
2. the sampled token decodes exactly to `eos_token`;
3. an error occurs.

Finally:

```python
return tokenizer.decode(generated_tokens)
```

The returned text includes the original prompt because `generated_tokens`
started as a copy of `prompt_tokens`.

---

## 10. Training vs Generation

Training and generation use the same Transformer LM but in different modes.

| Topic | Training | Generation |
|---|---|---|
| Input | Random sampled batches from `train.dat` | Prompt tokens plus sampled tokens |
| Shape | `(B, S)` | `(1, S_current)` |
| Output used | Logits at all positions | Logits only at final position |
| Target ids | Known shifted targets | No target ids |
| Loss | Cross entropy | None |
| Gradients | Yes | No, uses `torch.no_grad()` |
| Optimizer step | Yes | No |
| Model mode | `model.train()` | `model.eval()` |
| Token choice | Determined by dataset targets | Sampled from probability distribution |
| Context | Random fixed-length windows | Last `context_length` generated tokens |

Training teaches the model.

Generation asks the trained model to continue a prompt.

---

## 11. Important Invariants

Generation relies on these invariants:

1. The tokenizer used for generation must match the tokenizer used to produce
   the training data.
2. The checkpoint architecture must match the reconstructed `transformer_lm`.
3. `vocab_size` must match both tokenizer vocabulary and LM head output size.
4. Prompt text must encode to at least one token.
5. Input tensors must use integer token ids and dtype `torch.long`.
6. Model and input tensors must be on the same device.
7. `temperature` must be greater than 0.
8. `top_p` must be in `(0, 1]`.
9. Probabilities must be renormalized after top-p filtering.
10. The current implementation recomputes the full context each step and does
    not use KV cache.

---

## 12. Example Command

Example generation command:

```sh
uv run python cs336_basics/generate.py \
  --checkpoint artifacts/checkpoints/checkpoint_final_6000.pt \
  --vocab artifacts/tokenizers/owt_bpe_vocab32000_eot/vocab.pkl \
  --merges artifacts/tokenizers/owt_bpe_vocab32000_eot/merges.pkl \
  --prompt "Once upon a time" \
  --max_tokens 128 \
  --temperature 0.8 \
  --top_p 0.9 \
  --device auto
```

The exact paths depend on where tokenizer artifacts and checkpoints were saved.

---

## 13. Summary

Generation turns a prompt into text by repeatedly sampling the next token from
a trained language model.

The key ideas are:

- The prompt is encoded with the same tokenizer used during training.
- The model is reconstructed from checkpoint/config and loaded in eval mode.
- Each step uses the last `context_length` tokens as context.
- The model returns logits for every position, but generation uses only the
  final position.
- Temperature changes the sharpness of the distribution.
- Top-p sampling removes low-probability tail tokens and renormalizes.
- The sampled token is appended to the context.
- The final token sequence is decoded back into text.
