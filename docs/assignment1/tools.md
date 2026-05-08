# Utility Functions and Data Files

This note explains the helper code around the Assignment 1 training pipeline.

The main model, tokenizer, and generation code do the core machine learning
work. The utility code makes the pipeline usable: it prepares tokenized data
files, samples batches, records experiment metadata, and saves/restores
training state.

The high-level utility flow is:

```text
raw text files
  ↓
tokenize_dataset.py
  ↓
train.dat / valid.dat / meta.json
  ↓
train.py
  ↓
np.memmap data loading
  ↓
data_loading(...)
  ↓
input_ids / target_ids
  ↓
training loop
  ↓
metrics logging + checkpointing
```

The implementation lives mainly in:

| File | Purpose |
|---|---|
| `cs336_basics/scripts/tokenize_dataset.py` | Convert raw text files into token-id `.dat` files |
| `cs336_basics/trainer/data_loading.py` | Sample random input/target batches from token-id arrays |
| `cs336_basics/check_pointing.py` | Save and load model/optimizer checkpoints |
| `cs336_basics/trainer/experiment_logger.py` | Write local config, metrics, and summary files |
| `cs336_basics/train.py` | Uses these utilities inside the full training loop |

Document map:

| Section | Core content |
|---|---|
| [1. Utility Layer Role](#1-utility-layer-role) | What these helper functions do in the project |
| [2. File and Artifact Map](#2-file-and-artifact-map) | Files created and consumed by the pipeline |
| [3. Function Map](#3-function-map) | Table of helper functions and their data operations |
| [4. Dataset Tokenization Utilities](#4-dataset-tokenization-utilities) | How raw text becomes `train.dat` and `valid.dat` |
| [5. Data Loading During Training](#5-data-loading-during-training) | How random input/target batches are sampled |
| [6. Experiment Logging](#6-experiment-logging) | Config, metrics, CSV/JSONL logs, and summary files |
| [7. Checkpointing](#7-checkpointing) | Saving/restoring model state, optimizer state, and iteration |
| [8. How These Utilities Connect in Training](#8-how-these-utilities-connect-in-training) | End-to-end training-side utility flow |
| [9. Important Invariants](#9-important-invariants) | Assumptions that keep files and training state consistent |
| [10. Summary](#10-summary) | Compact recap |

---

## 1. Utility Layer Role

The utility layer does not define the Transformer architecture and does not
train the tokenizer itself.

Instead, it handles the practical data and file operations around training:

```text
prepare token-id files
open token-id files efficiently
sample random batches
record run configuration
append metrics
save checkpoints
resume checkpoints
```

This makes the main training loop smaller and more inspectable.

---

## 2. File and Artifact Map

The training pipeline reads and writes several files.

| Artifact | Created by | Used by | Purpose |
|---|---|---|---|
| `vocab.pkl` | tokenizer training | `Tokenizer.from_files` | Token id to byte sequence vocabulary |
| `merges.pkl` | tokenizer training | `Tokenizer.from_files` | Ordered BPE merge rules |
| tokenizer `meta.json` | tokenizer training | `tokenize_dataset.py` | Special tokens and tokenizer metadata |
| `train.dat` | `tokenize_dataset.py` | `train.py` | Token ids for training split |
| `valid.dat` | `tokenize_dataset.py` | `train.py` | Token ids for validation split |
| tokenized data `meta.json` | `tokenize_dataset.py` | `train.py`, logger | Dataset/tokenizer metadata |
| `config.json` | `LocalExperimentLogger` | human inspection | Run configuration and metadata |
| `metrics.jsonl` | `LocalExperimentLogger` | human/scripts | One metric record per line |
| `metrics.csv` | `LocalExperimentLogger` | spreadsheet/analysis | Tabular metric log |
| `summary.json` | `LocalExperimentLogger` | human inspection | Final run summary |
| `checkpoint_*.pt` | `save_checkpoint` | resume/generation | Model and optimizer state |

The `.dat` files contain only token ids. They do not store raw text, token
strings, byte sequences, or positions.

---

## 3. Function Map

| Function / object | Input | Operation on data/state | Output |
|---|---|---|---|
| `tokenize_dataset.parse_args()` | CLI args | Parses tokenizer/data/output paths and dtype | args namespace |
| `load_special_tokens(tokenizer_dir)` | tokenizer artifact directory | Reads tokenizer `meta.json` if present | `list[str]` |
| `count_lines(path)` | text file path | Counts lines for progress bars | integer line count |
| `choose_dtype(dtype_name, vocab_size)` | dtype name and vocab size | Checks whether dtype can store all token ids | `np.dtype` |
| `encode_txt_as_memmap(tokenizer, input_path, output_path, dtype)` | tokenizer and raw text file | Counts tokens, creates memmap, writes token ids | total token count |
| `run_sanity_check(tokenizer, input_path, sanity_chars)` | tokenizer and text sample | Checks encode/decode round trip on a small prefix | printed sanity report |
| `data_loading(dataset, batch_size, context_length, device)` | token-id array | Samples random shifted input/target sequences | `(inputs, targets)` tensors |
| `save_checkpoint(model, optimizer, iteration, out, model_config)` | training state | Serializes model/optimizer state and metadata | `.pt` file |
| `load_checkpoint(src, model, optimizer)` | checkpoint path | Loads model and optimizer state | saved iteration |
| `slugify(text)` | arbitrary text | Converts text into filesystem/log friendly slug | slug string |
| `infer_assignment_name()` | current file path | Finds assignment directory name from parent paths | assignment slug |
| `infer_dataset_name(data_dir, data_meta)` | data directory and metadata | Uses tokenizer dir or data dir to name dataset | dataset slug |
| `build_run_name(args, assignment_name, dataset_name)` | args and inferred names | Builds stable descriptive run name | run name string |
| `get_model_config(args)` | training args | Extracts model architecture config | dict |
| `get_optimizer_config(args)` | training args | Extracts optimizer config | dict |
| `get_training_config(args)` | training args | Extracts loop config | dict |
| `write_json(path, data)` | file path and dict | Writes pretty JSON | JSON file |
| `LocalExperimentLogger(...)` | args, device, metadata, model config | Creates local log files and config | logger object |
| `LocalExperimentLogger.add_dataset_info(...)` | dataset paths/counts and parameter count | Updates `config.json` with dataset/model info | updated config |
| `LocalExperimentLogger.log_metric(record)` | metric dict | Appends JSONL/CSV rows and tracks latest train/val | metric files |
| `LocalExperimentLogger.write_summary(...)` | final training status | Writes final run summary | `summary.json` |

---

## 4. Dataset Tokenization Utilities

`scripts/tokenize_dataset.py` converts raw text into compact token-id files.

The flow is:

```text
load tokenizer artifacts
  ↓
load special tokens from tokenizer meta.json
  ↓
choose output dtype
  ↓
sanity-check encode/decode
  ↓
encode train text into train.dat
  ↓
encode valid text into valid.dat
  ↓
write tokenized-data meta.json
```

### 4.1 Loading Special Tokens

Special tokens are recovered from tokenizer metadata:

```python
meta_path = tokenizer_dir / "meta.json"
return meta.get("special_tokens", [])
```

If `meta.json` does not exist, the script uses an empty list.

This matters because special tokens such as `<|endoftext|>` need to be
preserved as atomic token ids during dataset encoding.

---

### 4.2 Choosing dtype

The tokenized dataset is stored as a flat numeric array.

For a tokenizer with:

```text
vocab_size <= 65536
```

`uint16` can store all token ids:

```text
uint16 max = 65535
```

The helper checks:

```python
if vocab_size - 1 > max_value:
    raise ValueError(...)
```

This prevents silently writing token ids that cannot fit in the chosen dtype.

---

### 4.3 Encoding Text as Memmap

`encode_txt_as_memmap` uses two passes over the input text file.

First pass:

```text
read each line
  ↓
tokenizer.encode(line)
  ↓
accumulate total token count
```

This determines the exact output array length.

Second pass:

```text
create np.memmap(output_path, mode="w+", shape=(total_tokens,))
  ↓
read each line again
  ↓
encode line into token ids
  ↓
write ids into the correct array slice
  ↓
flush memmap
```

The output is a flat array:

```text
[id0, id1, id2, id3, ..., idN]
```

There are no document boundaries except whatever special tokens are already
present in the encoded text.

---

## 5. Data Loading During Training

`data_loading` samples mini-batches from a token-id array.

Input:

```text
dataset: token-id array, usually np.memmap
batch_size: B
context_length: S
```

For each batch item:

```python
start_idx = torch.randint(0, dataset_len - context_length, (1,)).item()
input_seq = dataset[start_idx : start_idx + context_length]
target_seq = dataset[start_idx + 1 : start_idx + context_length + 1]
```

Output:

```text
inputs:  (B, S)
targets: (B, S)
```

Example:

```text
dataset: [10, 11, 12, 13, 14, 15]
start:   1
S:       3

input:   [11, 12, 13]
target:  [12, 13, 14]
```

The returned tensors are moved to the requested device:

```python
inputs = inputs.to(device=device)
targets = targets.to(device=device)
```

This keeps the training loop simple because batches are already on the same
device as the model.

---

## 6. Experiment Logging

`LocalExperimentLogger` records local run information even when W&B is disabled
or unavailable.

When initialized, it creates:

```text
config.json
metrics.jsonl
metrics.csv
summary.json
```

inside the checkpoint directory.

### 6.1 Config File

`config.json` records:

```text
run_name
started_at
host
assignment
dataset_name
device_requested
device_resolved
data_dir
save_ckp_path
model_config
optimizer_config
training_config
wandb config
tokenized_data_meta
all_args
```

Later, `add_dataset_info` adds:

```text
train_data_path
val_data_path
train_tokens
val_tokens
model_total_parameters
```

This makes a run reproducible from its saved configuration.

---

### 6.2 Metrics Files

Each metric record is augmented with a timestamp:

```python
record = {
    "time": datetime.now().isoformat(),
    **record,
}
```

Then it is written to:

```text
metrics.jsonl
metrics.csv
```

JSONL is useful for scripts and structured records.

CSV is useful for quick spreadsheet-style inspection.

The logger also stores:

```text
latest_train
latest_val
```

in memory so the final summary can include the most recent train and validation
metrics.

---

### 6.3 Summary File

At the end of training, `write_summary` records:

```text
run_name
status
started_at
finished_at
elapsed_seconds
completed_iter
total_params
final_checkpoint_path
latest_train
latest_val
metrics_path
metrics_csv_path
config_path
```

This is the final compact report for the run.

---

## 7. Checkpointing

Checkpoints store enough state to resume training.

`save_checkpoint` writes:

```python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "iteration": iteration,
}
```

If `model_config` is passed, it also writes:

```text
model_config
```

This is useful for generation because the model architecture must be
reconstructed before weights can be loaded.

### 7.1 Model State

`model.state_dict()` contains all trainable parameters and registered buffers.

Examples:

```text
token_embedding.weight
layers.0.ln1.g_weight
layers.0.attn.w_q.weight
...
```

These tensors define the trained model.

---

### 7.2 Optimizer State

`optimizer.state_dict()` stores optimizer-specific state.

For AdamW, this includes moment estimates such as:

```text
m
v
t
```

Restoring optimizer state matters when resuming training. If only model weights
were restored, the optimizer would lose its momentum/adaptive statistics.

---

### 7.3 Loading a Checkpoint

`load_checkpoint` does:

```python
ckp = torch.load(src, map_location="cpu")
model.load_state_dict(ckp["model_state"])
optimizer.load_state_dict(ckp["optimizer_state"])
return ckp["iteration"]
```

The returned iteration becomes `start_iter` in the training loop.

The checkpoint is loaded on CPU first. The training script has already moved
the model to the target device; PyTorch handles parameter copying when loading
state dicts.

---

## 8. How These Utilities Connect in Training

The utility functions connect in this order:

```text
before training:
    tokenize_dataset.py
        raw text → train.dat / valid.dat / meta.json

training startup:
    load_data_meta(data_dir)
    get_model_config(args)
    LocalExperimentLogger(...)
    get_dataset_memmap(train.dat)
    get_dataset_memmap(valid.dat)
    optionally load_checkpoint(...)

each training step:
    data_loading(...)
    model forward / loss / backward
    LocalExperimentLogger.log_metric(...)
    optionally save_checkpoint(...)

after training:
    save final checkpoint
    LocalExperimentLogger.write_summary(...)
```

The important boundary is:

```text
tokenize_dataset.py prepares files
train.py consumes files
```

Training should not re-tokenize raw text. It reads the already-tokenized
`.dat` files.

---

## 9. Important Invariants

These utilities rely on several invariants:

1. `train.dat` and `valid.dat` must use the same tokenizer vocabulary as the
   model's `vocab_size`.
2. The selected dtype must be large enough for all token ids.
3. `data_loading` needs at least `context_length + 1` tokens available.
4. `input_seq` and `target_seq` must be shifted by exactly one token.
5. Model and batch tensors must be on the same device.
6. Checkpoint model architecture must match the model object being loaded.
7. Resume training should restore both model state and optimizer state.
8. Local logging can be disabled, so training logic should not depend on log
   files existing.
9. `meta.json`, `config.json`, and `summary.json` are metadata files for humans
   and scripts; they are not used by the model forward pass.

---

## 10. Summary

The utility layer is the glue around the model.

It handles:

- converting raw text into token-id memmap files;
- sampling random shifted batches;
- saving and loading checkpoints;
- recording experiment configuration;
- appending training and validation metrics;
- writing final run summaries.

These helpers keep the core model code focused on tensor computation while the
training scripts manage files, metadata, and reproducibility.
