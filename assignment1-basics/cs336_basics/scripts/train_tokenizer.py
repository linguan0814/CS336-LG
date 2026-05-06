from pathlib import Path
import sys
import json
import pickle
from datetime import datetime
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs336_basics.train_bpe import train_bpe


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "owt_train.txt"

# config
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
RUN_NAME = f"owt_bpe_vocab{VOCAB_SIZE}_eot-im"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "tokenizers" / RUN_NAME
VOCAB_PATH = OUTPUT_DIR / "vocab.pkl"
MERGES_PATH = OUTPUT_DIR / "merges.pkl"
META_PATH = OUTPUT_DIR / "meta.json"


print("开始训练 tokenizer...")
print(f"input: {INPUT_PATH}")
print(f"vocab size: {VOCAB_SIZE}")
print(f"special tokens: {SPECIAL_TOKENS}")
print(f"output dir: {OUTPUT_DIR}")

start_time = perf_counter()
vocab, merges = train_bpe(
    input_path=INPUT_PATH,
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
)
elapsed_seconds = perf_counter() - start_time

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with VOCAB_PATH.open("wb") as f:
    pickle.dump(vocab, f)

with MERGES_PATH.open("wb") as f:
    pickle.dump(merges, f)

longest_token = max(vocab.values(), key=len)

meta = {
    "input_path": str(INPUT_PATH),
    "vocab_size": VOCAB_SIZE,
    "special_tokens": SPECIAL_TOKENS,
    "num_vocab_items": len(vocab),
    "num_merges": len(merges),
    "longest_token_len": len(longest_token),
    "longest_token_preview": repr(longest_token),
    "elapsed_seconds": elapsed_seconds,
    "created_at": datetime.now().isoformat(),
    "notes": "OpenWebText train split",
}

with META_PATH.open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"训练完成，保存至: {OUTPUT_DIR}")
print(f"vocab size: {len(vocab)}")
print(f"num merges: {len(merges)}")
print(f"longest token: {repr(longest_token)}")
print(f"longest token len: {len(longest_token)}")
print(f"elapsed: {elapsed_seconds / 60:.2f} min")
