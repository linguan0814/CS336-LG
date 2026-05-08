# Byte-level BPE Tokenizer

This note explains the tokenizer part of Assignment 1.

The goal is to understand how raw text becomes integer token ids before it is
fed into a Transformer language model.

In this project, the tokenizer is a **byte-level BPE tokenizer**:

```text
text
  ↓
Unicode string
  ↓
UTF-8 bytes
  ↓
pre-tokenization
  ↓
BPE merges
  ↓
token ids
```

The implementation lives mainly in:

| File | Purpose |
|---|---|
| `cs336_basics/train_bpe.py` | Train byte-level BPE vocab and merge rules |
| `cs336_basics/train_bpe_simple.py` | Baseline full-rescan BPE trainer |
| `cs336_basics/tokenizer.py` | Runtime tokenizer: encode/decode text |
| `cs336_basics/scripts/train_tokenizer.py` | Train and save tokenizer artifacts |
| `cs336_basics/scripts/tokenize_dataset.py` | Encode raw datasets into `.dat` token-id files |
| `cs336_basics/pretokenization_example.py` | CS336 chunk-boundary example for parallel pre-tokenization |

Document map:

| Section | Core content |
|---|---|
| [1. Why Text Needs a Tokenizer](#1-why-text-needs-a-tokenizer) | Why a Transformer needs integer token ids instead of raw strings |
| [2. From Text to Bytes](#2-from-text-to-bytes) | Unicode code points, UTF-8 bytes, and why byte-level tokenization can cover arbitrary text |
| [3. Character, Word, Byte, and Subword Tokenization](#3-character-word-byte-and-subword-tokenization) | Tradeoffs between tokenization granularities and why BPE is a useful compromise |
| [4. Training a Byte-level BPE Tokenizer](#4-training-a-byte-level-bpe-tokenizer) | How raw text becomes `vocab` and `merges`, including chunked large-file processing and the function table |
| [5. Baseline vs Optimized BPE Training](#5-baseline-vs-optimized-bpe-training) | Why the simple trainer is slow and how the optimized trainer updates only affected words |
| [6. Encoding with the Learned Tokenizer](#6-encoding-with-the-learned-tokenizer) | How text is converted into token ids, including special-token handling and the encode function table |
| [7. Decoding](#7-decoding) | How token ids are converted back into text and what round-trip correctness means |
| [8. Encoding Large Datasets](#8-encoding-large-datasets) | How trained tokenizer artifacts are used to write `train.dat` and `valid.dat` |
| [9. Chunking and Parallel Pre-tokenization](#9-chunking-and-parallel-pre-tokenization) | How CS336 suggests splitting large files safely around document boundaries |
| [10. Relationship to the Transformer](#10-relationship-to-the-transformer) | How tokenizer `vocab_size` connects to model embeddings and output logits |
| [11. Important Invariants](#11-important-invariants) | Properties that must stay true for training, encoding, and decoding |
| [12. Commands Used in This Project](#12-commands-used-in-this-project) | Commands for training the tokenizer, tokenizing data, and running tokenizer tests |
| [13. Summary](#13-summary) | Compact recap of the tokenizer pipeline |

---

## 1. Why Text Needs a Tokenizer

A Transformer does not directly understand characters, words, or strings.

The model input is an integer tensor:

```text
input_ids: (batch_size, sequence_length)
```

For example:

```text
"Hello" → [15496]
```

or, depending on the tokenizer:

```text
"Hello" → [39, 68, 75, 75, 78]
```

The tokenizer is the bridge between human-readable text and model-readable
integer ids.

It must solve three problems:

1. How to represent arbitrary text from the real world.
2. How to compress text into a useful sequence of subword tokens.
3. How to decode token ids back into the original text.

The byte-level BPE tokenizer solves these by starting from bytes, then learning
frequent byte-sequence merges from the training corpus.

---

## 2. From Text to Bytes

### 2.1 Unicode

Modern text is usually represented by Unicode.

Unicode assigns each character an abstract code point:

```text
"A"  → U+0041
"é"  → U+00E9
"你" → U+4F60
"🙃" → U+1F643
```

A code point is not yet the bytes stored in memory or on disk. It is an
abstract number that names a character.

This distinction matters because a language model ultimately needs integers,
and files ultimately store bytes.

---

### 2.2 UTF-8 Encoding

UTF-8 is one way to encode Unicode code points into bytes.

It uses a variable number of bytes per character:

| Text | UTF-8 bytes |
|---|---|
| `A` | `41` |
| `é` | `c3 a9` |
| `你` | `e4 bd a0` |
| `🙃` | `f0 9f 99 83` |

In Python:

```python
"🙃".encode("utf-8")
```

returns:

```text
b'\xf0\x9f\x99\x83'
```

This is the key reason CS336 uses a byte-level tokenizer: if the base vocabulary
contains all 256 possible bytes, then every valid UTF-8 string can be
represented.

The tokenizer does not need an `<unk>` token for normal text, because every
character can be decomposed into bytes.

---

## 3. Character, Word, Byte, and Subword Tokenization

There are several possible tokenization strategies.

| Strategy | Example | Problem |
|---|---|---|
| Character-level | `hello → h e l l o` | Long sequences |
| Word-level | `hello world → hello world` | Huge vocab and unknown words |
| Byte-level | `hello → 68 65 6c 6c 6f` | Very long sequences |
| Subword BPE | `hello world → hello Ġworld` | Good compromise |

BPE is a compromise:

- frequent words or pieces become single tokens;
- rare words can still be represented by smaller pieces;
- byte-level fallback guarantees coverage.

For language modeling, this is important because the model context length is
counted in tokens, not characters. A better tokenizer can represent the same
text using fewer tokens.

---

## 4. Training a Byte-level BPE Tokenizer

Tokenizer training starts from a raw text corpus and produces `vocab` and
`merges`.

This chapter is about **training the tokenizer**, not training the Transformer.
BPE training is a corpus statistics algorithm: it reads text, counts repeated
byte patterns, and turns the most frequent patterns into vocabulary entries.

If we only used the 256 byte tokens, the tokenizer could represent any text,
but the token sequences would be very long. BPE keeps the byte-level guarantee,
then learns frequent byte sequences from the corpus so common text can be
represented with fewer tokens.

The output is:

```text
vocab:  token id → byte sequence
merges: ordered list of byte-pair merge rules
```

This distinction matters:

```text
BPE training:
    raw text → vocab / merges

LM training:
    token ids → model parameters
```

The tokenizer must be trained first because the language model needs a fixed
`vocab_size` before its embedding table and output head can be initialized.

---

### 4.1 Core Training Data Flow

The most important point is this:

**Large files are read in chunks, but BPE is not trained separately on each
chunk.**

Chunking is only a memory strategy for building one global counter. The BPE
merge loop starts only after all chunks have contributed to that global counter.

The high-level flow in this implementation is:

```text
raw text corpus
  ↓
stream file in text chunks
  ↓
for each chunk:
    combine with previous carry text
    take safe_text, keep new carry
    remove/split around special tokens
    run regex pre-tokenization
    encode each pre-token to UTF-8 byte tuple
    update global cnt_pretokens
  ↓
after the whole file is consumed:
    process final carry
    build words / word_counts from global cnt_pretokens
    build global pair_counts / pair_to_words / pair_heap
  ↓
while len(vocab) < vocab_size:
    choose globally most frequent adjacent pair
    append pair to merges
    add concatenated bytes to vocab
    update only words affected by that pair
  ↓
return vocab and merges
```

So the boundary between "reading chunks" and "training BPE merges" is:

```text
global cnt_pretokens is complete
```

At that moment, the algorithm no longer cares which chunk a pre-token came
from. It has a corpus-level count of all pre-token byte tuples.

---

### 4.2 Training Functions and Data Operations

| Function / object | Input | Operation on data | Output |
|---|---|---|---|
| `train_bpe(input_path, vocab_size, special_tokens, show_progress)` | Path to raw text corpus and training config | Orchestrates the full BPE training pipeline | `(vocab, merges)` |
| `_init_vocab(vocab, special_token)` | Empty dict and special token strings | Encodes special tokens to UTF-8 bytes, then adds all 256 single-byte values | Initial `dict[int, bytes]` vocabulary |
| chunk reading loop inside `train_bpe` | Raw text file | Reads `CHUNK_SIZE` text chunks and carries over the last `TAIL_SIZE` characters | `safe_text` pieces plus final `carry` |
| `pre_tokenization(s, special_token)` | A safe text span | Splits around special tokens, skips them for training, then applies GPT-2-style regex | Iterator of pre-token strings |
| `word_2_byte(word)` | One pre-token string | Encodes with UTF-8 and wraps every byte as a `bytes` object | `tuple[bytes, ...]` |
| `cnt_pretokens` | Pre-token byte tuples | Accumulates counts across all chunks | `Counter[tuple[bytes, ...]]` |
| `_adjacent_pairs(word)` | Current byte-token list for one pre-token | Lists neighboring byte-token pairs | `list[tuple[bytes, bytes]]` |
| `pair_counts` | All counted pre-token byte tuples | Counts adjacent pairs weighted by pre-token frequency | `Counter[tuple[bytes, bytes]]` |
| `pair_to_words` | Adjacent pairs and word ids | Records which current words contain each pair | `dict[pair, set[word_id]]` |
| `pair_heap` | Pair counts | Keeps candidate pairs ordered by count and tie-break priority | Heap of pair candidates |
| `_merge_word(word, merge_pair, new_token)` | One affected word and selected pair | Replaces non-overlapping occurrences of the selected adjacent pair | Updated `list[bytes]` |
| `_ReversePair` | Pair used inside heap item | Reverses pair ordering so heap tie-breaking matches the expected max lexicographic pair behavior | Comparable wrapper for heap entries |

The important intermediate data objects are:

| Object | Meaning | Shape / type |
|---|---|---|
| `vocab` | Token id to byte sequence | `dict[int, bytes]` |
| `merge_rule` / `merges` | Ordered learned BPE rules | `list[tuple[bytes, bytes]]` |
| `cnt_pretokens` | Corpus-level pre-token counts after chunk processing | `Counter[tuple[bytes, ...]]` |
| `words` | Mutable current byte-tokenization of each unique pre-token | `dict[int, list[bytes]]` |
| `word_counts` | Frequency of each unique pre-token | `dict[int, int]` |
| `pair_counts` | Current corpus-level adjacent-pair counts | `Counter[tuple[bytes, bytes]]` |
| `pair_to_words` | Reverse index from pair to words containing it | `dict[tuple[bytes, bytes], set[int]]` |

---

### 4.3 What BPE Learns

BPE stands for **Byte-Pair Encoding**.

In this assignment, BPE learns two artifacts:

```text
vocab:  dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

The vocabulary maps token ids to byte sequences:

```text
0     → b"<|endoftext|>"
1     → b"\x00"
2     → b"\x01"
...
257   → b"t"
...
1000  → b" the"
```

The merge list stores the learned merge rules in order:

```text
(b"t", b"h")
(b"th", b"e")
(b" ", b"the")
...
```

The order matters. During encoding, the tokenizer repeatedly applies the
highest-priority available merge, where priority means earliest in the learned
merge list.

So BPE training is not directly training a neural network. It is learning:

1. which byte sequences should become vocabulary entries;
2. in what order those byte sequences should be merged during encoding.

---

### 4.4 Initial Vocabulary

The initial vocabulary contains:

1. special tokens, such as `<|endoftext|>`;
2. all 256 possible byte values.

In `train_bpe.py`:

```python
def _init_vocab(vocab: dict, special_token: list):
    special_token_encoded = [s.encode("UTF-8") for s in special_token]
    ...
    for i in range(256):
        init_str = bytes([i])
        if init_str not in vocab.values():
            vocab[idx] = init_str
```

So for:

```text
special_tokens = ["<|endoftext|>"]
```

the vocabulary starts with:

```text
b"<|endoftext|>"
b"\x00"
b"\x01"
...
b"\xff"
```

Then BPE adds newly merged byte sequences until `len(vocab) == vocab_size`.

---

### 4.5 Special Tokens

Special tokens are not normal text pieces.

In Assignment 1, `<|endoftext|>` is used as a document boundary marker.

Special tokens need two different behaviors:

| Phase | Behavior |
|---|---|
| BPE training | Use special tokens as boundaries; do not merge them with nearby text |
| Encoding | Preserve special tokens as single token ids |

In `train_bpe.py`, special tokens are removed before normal pre-tokenization:

```text
"hello<|endoftext|>world"
  ↓
"hello" and "world" are counted
<|endoftext|> is not merged into normal tokens
```

This prevents bad vocabulary entries such as:

```text
b"hello<|endoftext|>"
b"<|endoftext|>world"
```

The tests check this behavior in `test_train_bpe_special_tokens`.

At runtime, `Tokenizer.encode` does preserve special tokens:

```text
"hello<|endoftext|>world"
  ↓
[ids for "hello"] + [id for "<|endoftext|>"] + [ids for "world"]
```

Overlapping special tokens are handled by sorting from longest to shortest
before regex splitting. This makes:

```text
<|endoftext|><|endoftext|>
```

match as one special token if that longer special token is registered.

---

### 4.6 Pre-tokenization

Before BPE counts byte pairs, the text is split into pre-tokens using a
GPT-2-style regex:

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This regex roughly separates:

| Pattern part | Meaning |
|---|---|
| `'(?:[sdmt]|ll|ve|re)` | contractions like `'s`, `'re`, `'ll` |
| ` ?\p{L}+` | optional leading space plus letters |
| ` ?\p{N}+` | optional leading space plus numbers |
| ` ?[^\s\p{L}\p{N}]+` | optional leading space plus punctuation/symbols |
| `\s+(?!\S)` | trailing whitespace |
| `\s+` | other whitespace |

The optional leading space is important.

It lets common word pieces include their preceding space:

```text
" hello" → one pre-token
"hello"  → another pre-token
```

This is why GPT-style tokenizers often have separate tokens for `"hello"` and
`" hello"`.

---

### 4.7 Convert Pre-tokens to Bytes

Each pre-token is encoded to UTF-8 and split into single-byte pieces.

In `train_bpe.py`:

```python
def word_2_byte(word: str) -> tuple[bytes, ...]:
    word_decoded = list(word.encode("UTF-8"))
    word_byte = [bytes([b]) for b in word_decoded]
    return tuple(word_byte)
```

Example:

```text
"hi" → (b"h", b"i")
"你" → (b"\xe4", b"\xbd", b"\xa0")
"🙃" → (b"\xf0", b"\x9f", b"\x99", b"\x83")
```

BPE merge rules operate on these byte pieces, not directly on Python
characters.

---

### 4.8 The Large-file Chunking Stage

For a small corpus, the simplest implementation can read the full file, run
pre-tokenization, and then start BPE merging.

For a large corpus like OpenWebText, reading the whole file into memory is not
ideal. The optimized trainer therefore reads the file in chunks:

```python
CHUNK_SIZE = 4 * 1024 * 1024
TAIL_SIZE = 4096
carry = ""
```

Each loop reads a new chunk and prepends the leftover `carry` from the previous
loop:

```text
buffer = carry + chunk
safe_text = buffer[:-TAIL_SIZE]
carry = buffer[-TAIL_SIZE:]
```

Only `safe_text` is pre-tokenized immediately. The last `TAIL_SIZE` characters
are delayed until the next chunk so that a pre-token near the boundary is less
likely to be cut in half.

The chunking stage does this:

```text
text chunk
  ↓
safe_text plus carry logic
  ↓
split around special tokens
  ↓
regex pre-tokenization
  ↓
UTF-8 byte tuples
  ↓
update global cnt_pretokens
```

The key object produced by this stage is:

```text
cnt_pretokens: Counter[tuple[bytes, ...]]
```

Example:

```text
cnt_pretokens[(b"t", b"h", b"e")] += 1
cnt_pretokens[(b" ", b"c", b"a", b"t")] += 1
```

This counter is global across the whole corpus. If the same pre-token appears
in many different chunks, all of those occurrences are accumulated into the
same counter entry.

So the chunking stage stops at this level:

```text
raw corpus → global counts of pre-token byte tuples
```

It does **not** compute a separate `vocab` or `merges` for each chunk.

Only after the file has been fully processed, including the final `carry`, does
the trainer move into the BPE merge phase:

```python
for word_id, (pretoken, cnt) in enumerate(cnt_pretokens.items()):
    word = list(pretoken)
    words[word_id] = word
    word_counts[word_id] = cnt
    for pair in _adjacent_pairs(word):
        pair_counts[pair] += cnt
        pair_to_words.setdefault(pair, set()).add(word_id)
```

At this point the algorithm has a corpus-level view of pair frequencies.

---

### 4.9 Count Adjacent Pairs

After pre-tokenization, the trainer counts how often each pre-token appears:

```text
cnt_pretokens[(b"t", b"h", b"e")] = 12345
cnt_pretokens[(b" ", b"t", b"h", b"e")] = 67890
```

Then it counts adjacent pairs, weighted by pre-token frequency:

```text
(b"t", b"h")
(b"h", b"e")
```

If `(b"t", b"h")` appears inside a pre-token that occurs 10,000 times, then that
pair contributes 10,000 to the pair count.

---

### 4.10 Merge the Best Pair

At each BPE step:

1. find the most frequent adjacent pair;
2. create a new token by concatenating the bytes;
3. append the pair to `merges`;
4. add the new byte sequence to `vocab`;
5. replace every occurrence of that pair inside all pre-token byte sequences.

Example:

```text
Before:
(b"t", b"h", b"e")

Merge:
(b"t", b"h") → b"th"

After:
(b"th", b"e")
```

If the next best merge is `(b"th", b"e")`, then:

```text
(b"th", b"e") → b"the"
```

This is how frequent byte sequences become larger subword tokens.

When two pairs have the same frequency, the implementation follows the CS336
test expectation by using lexicographic tie-breaking on the pair.

---

## 5. Baseline vs Optimized BPE Training

This repository contains two BPE trainers.

### 5.1 Baseline Full-rescan Version

`train_bpe_simple.py` is the direct implementation.

For every merge step, it:

1. scans all pre-tokens;
2. rebuilds pair counts from scratch;
3. applies the selected merge;
4. repeats.

This is easy to understand but slow:

```text
num_merges × number_of_pretokens × average_pretoken_length
```

It is useful as a conceptual reference.

---

### 5.2 Optimized Version

`train_bpe.py` is the faster version used for larger datasets.

It keeps several data structures:

| Structure | Purpose |
|---|---|
| `words` | `word_id → current list[bytes]` |
| `word_counts` | `word_id → frequency` |
| `pair_counts` | adjacent pair frequency |
| `pair_to_words` | pair → set of word ids containing the pair |
| `pair_heap` | priority queue for choosing the best pair |

Instead of rescanning all words after each merge, it updates only words that
actually contain the selected pair.

The optimized flow is:

```text
initialize all pair counts
  ↓
push pairs into heap
  ↓
pop current most frequent pair
  ↓
find affected words from pair_to_words
  ↓
subtract old pair counts for affected words
  ↓
merge selected pair in those words
  ↓
add new pair counts for affected words
  ↓
push changed pairs back into heap
```

The heap can contain stale entries, so the trainer checks whether the popped
count still matches `pair_counts[pair]`. If it does not match, that heap entry
is ignored.

This is a common lazy-update priority queue pattern.

---

## 6. Encoding with the Learned Tokenizer

Encoding is the runtime process that turns a string into token ids.

After BPE training, the tokenizer has:

```text
vocab.pkl
merges.pkl
meta.json
```

`vocab.pkl` and `merges.pkl` are loaded into `Tokenizer`. `meta.json` is used
by scripts such as `tokenize_dataset.py` to recover the special token list.

Encoding does not learn new tokens. It only applies the vocabulary and merge
rules that were already learned during tokenizer training.

The key invariant is:

```text
every emitted token id must correspond to one byte sequence in vocab
```

---

### 6.1 Encode Data Flow

The main encode flow is:

```text
raw text string
  ↓
Tokenizer.encode(text)
  ↓
Tokenizer.pre_tokenization(text, special_tokens)
  ↓
for each pre-token:
      if special token:
          emit special token id
      else:
          encode_text(pre_token)
              ↓
           UTF-8 byte tuple
              ↓
           apply_merge(byte tuple)
              ↓
           lookup each merged byte sequence in byte_2_id_vocab
  ↓
list[int] token ids
```

For normal text, `encode_text` transforms one pre-token:

```text
pre-token string
  ↓
UTF-8 bytes
  ↓
apply learned merges
  ↓
lookup byte sequence ids
  ↓
token ids
```

Special tokens take a shorter path:

```text
special token string
  ↓
UTF-8 bytes
  ↓
byte_2_id_vocab lookup
  ↓
one special token id
```

This difference is intentional. Special tokens should not be split into bytes
and should not be affected by ordinary BPE merges.

---

### 6.2 Encoding Functions and Data Operations

| Function / object | Input | Operation on data | Output |
|---|---|---|---|
| `Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)` | Serialized vocab and merge files | Loads pickle files and normalizes ids and values to `int` and `bytes` | Initialized `Tokenizer` |
| `Tokenizer.__init__(vocab, merges, special_tokens)` | `dict[int, bytes]`, merge list, optional special tokens | Stores artifacts, builds inverse vocab, ranks merges, registers missing special tokens | Tokenizer object with lookup tables |
| `self.byte_2_id_vocab` | `vocab` | Inverts `id → bytes` into `bytes → id` | `dict[bytes, int]` |
| `self.merges_ranking` | Ordered `merges` list | Converts merge order into numeric priority | `dict[tuple[bytes, bytes], int]` |
| `Tokenizer.encode(text)` | Raw text string | Runs pre-tokenization, routes special tokens and normal text pieces, concatenates ids | `list[int]` |
| `Tokenizer.pre_tokenization(s, special_tokens)` | Raw string and special token list | Splits special tokens first, then applies GPT-2-style regex to non-special spans | `list[str]` pre-tokens |
| `Tokenizer.encode_text(pre_token)` | One non-special pre-token string | UTF-8 encodes, applies BPE merges, converts merged bytes to ids | `list[int]` |
| Local `word_2_byte` inside `encode_text` | One pre-token string | Encodes text as UTF-8 and wraps each byte | `tuple[bytes, ...]` |
| `Tokenizer.apply_merge(word_byte)` | Tuple/list of byte tokens | Repeatedly applies the highest-priority available merge | `list[bytes]` merged byte tokens |
| `Tokenizer.encode_iterable(iterable)` | Iterable of text chunks or lines | Calls `encode` chunk by chunk and yields ids lazily | Iterator of token ids |

The main lookup structures used during encoding are:

| Structure | Meaning | Used for |
|---|---|---|
| `vocab` | token id → bytes | Artifact identity and later decoding |
| `byte_2_id_vocab` | bytes → token id | Convert merged byte sequences into ids |
| `merges_ranking` | pair → merge priority | Decide which pair to merge first |
| `special_tokens` | strings that should stay atomic | Preserve markers such as `<|endoftext|>` |

---

### 6.3 Merge Priority During Encoding

The tokenizer stores merge priority as:

```python
self.merges_ranking = {merge: idx for idx, merge in enumerate(merges)}
```

Lower `idx` means higher priority.

During `apply_merge`, it repeatedly chooses the available pair with the best
ranking:

```python
bigram = min(
    word_pairs,
    key=lambda pair: self.merges_ranking.get(pair, float("inf")),
)
```

If no pair in the current word appears in `merges_ranking`, merging stops.

This reproduces the merge order learned during BPE training.

---

### 6.4 Example Encoding

Suppose the tokenizer learned:

```text
(b"t", b"h")  → b"th"
(b"th", b"e") → b"the"
```

Then:

```text
"the"
  ↓ UTF-8 bytes
(b"t", b"h", b"e")
  ↓ first merge
(b"th", b"e")
  ↓ second merge
(b"the",)
  ↓ vocab lookup
[token_id_for_b"the"]
```

For a Unicode character:

```text
"你"
  ↓ UTF-8 bytes
(b"\xe4", b"\xbd", b"\xa0")
  ↓ maybe some learned merges
[token ids]
```

Even if no merge exists for these bytes, the tokenizer can still emit the
single-byte ids because all 256 bytes are in the initial vocabulary.

---

### 6.5 Encoding Invariants

Encoding should preserve these invariants:

1. It never changes `vocab` or `merges`.
2. It never creates a byte sequence that is absent from `byte_2_id_vocab`.
3. Special tokens are emitted as atomic ids.
4. Normal text is first represented as UTF-8 bytes, so arbitrary Unicode text is
   representable.
5. The output is only token ids; it does not include raw text, byte offsets, or
   positions.

---

## 7. Decoding

Decoding is the inverse runtime process: it turns token ids back into text.

Compared with encoding, decoding is simpler because it does not need
pre-tokenization or BPE merge priority. The merge decisions have already been
encoded into the token ids.

---

### 7.1 Decode Data Flow

The decode flow is:

```text
list[int] token ids
  ↓
for each id:
    vocab[id] → bytes
  ↓
concatenate all byte sequences
  ↓
decode bytes with UTF-8
  ↓
text string
```

In `Tokenizer.decode`:

```python
byte_list = b"".join(self.vocab[id_] for id_ in ids)
return byte_list.decode("UTF-8", errors="replace")
```

The tokenizer does three data operations:

1. maps each token id back to its byte sequence;
2. concatenates all bytes;
3. decodes the full byte string as UTF-8.

---

### 7.2 Decoding Functions and Data Operations

| Function / object | Input | Operation on data | Output |
|---|---|---|---|
| `Tokenizer.decode(ids)` | `list[int]` token ids | Looks up each id in `vocab`, joins bytes, decodes UTF-8 | Python `str` |
| `self.vocab` | Token id | Maps id to the byte sequence learned or initialized during BPE training | `bytes` |
| `b"".join(...)` | Sequence of byte pieces | Reconstructs one continuous byte string | `bytes` |
| `bytes.decode("UTF-8", errors="replace")` | Continuous byte string | Converts UTF-8 bytes back into Unicode text; invalid sequences become replacement characters | Python `str` |

Decoding does not use:

| Not used in decode | Why |
|---|---|
| `merges` | Merge choices are already represented by token ids |
| `merges_ranking` | No merge decisions are made during decoding |
| `pre_tokenization` | Text boundaries are reconstructed from bytes, not re-tokenized |
| `byte_2_id_vocab` | Decode goes from id to bytes, not bytes to id |

---

### 7.3 Round-trip Meaning

For valid token sequences produced by this tokenizer:

```text
decode(encode(text)) == text
```

The tests check this for:

- empty strings;
- ASCII strings;
- Unicode strings;
- emoji;
- German text;
- TinyStories samples;
- special-token edge cases.

The `errors="replace"` argument means invalid UTF-8 byte sequences decode with
the Unicode replacement character. Normal encode/decode round trips should not
hit this case.

The reverse direction is not always guaranteed for arbitrary ids:

```text
encode(decode(ids)) == ids
```

Different token sequences can sometimes decode to the same text because one
sequence may use a larger merged token while another uses smaller byte-level
tokens.

For language-model usage, the important guarantee is the text round trip:

```text
text → encode → ids → decode → same text
```

---

## 8. Encoding Large Datasets

After training the tokenizer, the raw dataset is converted into token ids by
`scripts/tokenize_dataset.py`.

The flow is:

```text
load vocab.pkl and merges.pkl
  ↓
load special tokens from meta.json
  ↓
Tokenizer.from_files(...)
  ↓
encode train text
  ↓
write train.dat
  ↓
encode validation text
  ↓
write valid.dat
```

The `.dat` files contain only token ids.

They do not contain:

```text
raw text
token strings
byte sequences
positions
```

For vocab sizes below 65,536, `uint16` is enough:

```text
max uint16 = 65535
vocab_size = 10000 or 32000
```

The training pipeline later reads these files with `np.memmap`, so the whole
tokenized dataset does not need to be loaded into memory.

---

## 9. Chunking and Parallel Pre-tokenization

The CS336 handout highlights that pre-tokenization can be parallelized by
splitting a large file into chunks.

But chunking text is subtle:

1. A chunk boundary should not cut through a UTF-8 byte sequence.
2. A chunk boundary should not let text on different sides of
   `<|endoftext|>` merge together.
3. Special tokens should be removed before normal pre-tokenization.

`pretokenization_example.py` demonstrates one safe approach:

```text
open file as bytes
  ↓
choose approximate chunk boundaries
  ↓
move each boundary forward until <|endoftext|>
  ↓
decode each chunk
  ↓
pre-tokenize each chunk independently
```

The current optimized trainer in `train_bpe.py` uses a streaming single-process
approach instead:

```text
read text chunks
  ↓
keep a carry/tail from the previous chunk
  ↓
pre-tokenize only the safe prefix
  ↓
process the final carry at the end
```

This avoids reading the entire file at once, while keeping the implementation
smaller than a multiprocessing version.

---

## 10. Relationship to the Transformer

The tokenizer is trained before the language model.

The language model does not see text. It only sees token ids:

```text
raw text
  ↓ tokenizer.encode
token ids
  ↓ batch sampling
input_ids / target_ids
  ↓ Transformer LM
next-token prediction
```

The model's `vocab_size` must match the tokenizer vocabulary size.

If the tokenizer has:

```text
vocab_size = 32000
```

then the model needs:

```text
token embedding: 32000 rows
LM head output: 32000 logits
```

The tokenizer does not create positions. Token positions are handled inside the
model forward pass and RoPE logic.

---

## 11. Important Invariants

The tokenizer relies on several invariants:

1. `vocab` maps token ids to bytes.
2. `byte_2_id_vocab` is the inverse mapping from bytes to token ids.
3. All 256 single-byte values exist in the vocabulary.
4. `merges` is ordered from earliest learned merge to latest learned merge.
5. Special tokens are protected from normal BPE merging.
6. `decode(encode(text)) == text` for valid input text.
7. Tokenized `.dat` files store ids only, not text or positions.

These invariants are what make the tokenizer usable both for training and
generation.

---

## 12. Commands Used in This Project

Train a tokenizer:

```sh
uv run python cs336_basics/scripts/train_tokenizer.py
```

Tokenize the dataset:

```sh
uv run python cs336_basics/scripts/tokenize_dataset.py
```

Run tokenizer-related tests:

```sh
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py
```

The saved tokenizer artifacts are expected to look like:

```text
artifacts/tokenizers/<run_name>/
├── vocab.pkl
├── merges.pkl
└── meta.json
```

The tokenized dataset artifacts are expected to look like:

```text
artifacts/tokenized/<run_name>/
├── train.dat
├── valid.dat
└── meta.json
```

---

## 13. Summary

The byte-level BPE tokenizer is the first stage of the Assignment 1 language
model pipeline.

Its job is to turn arbitrary Unicode text into compact integer token sequences.

The key ideas are:

- Unicode text is encoded into UTF-8 bytes.
- The base vocabulary contains all 256 bytes, so arbitrary text is representable.
- BPE learns frequent byte-sequence merges from the corpus.
- Special tokens such as `<|endoftext|>` are protected from normal merging.
- Encoding applies the learned merges in priority order.
- Decoding concatenates byte sequences and decodes them as UTF-8.
- The Transformer receives token ids, not strings, bytes, or positions.

Once this tokenizer is trained and the dataset is encoded, the rest of the
pipeline can treat text as a long stream of integer ids for next-token
prediction.
