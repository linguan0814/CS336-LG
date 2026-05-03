import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "artifacts" / "tokenizers" / "tinystories_bpe_vocab10000_eot-im"
DEFAULT_TRAIN_TXT = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_VALID_TXT = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "tokenized" / "tinystories_bpe_vocab10000_eot-im"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize text datasets into token-id memmap files.")

    parser.add_argument("--tokenizer_dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--train_txt", type=Path, default=DEFAULT_TRAIN_TXT)
    parser.add_argument("--valid_txt", type=Path, default=DEFAULT_VALID_TXT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32", "int32", "int64"])
    parser.add_argument("--sanity_chars", type=int, default=1000)

    return parser.parse_args()


def load_special_tokens(tokenizer_dir: Path) -> list[str]:
    meta_path = tokenizer_dir / "meta.json"
    if not meta_path.exists():
        return []

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta.get("special_tokens", [])


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def choose_dtype(dtype_name: str, vocab_size: int) -> np.dtype:
    dtype = np.dtype(dtype_name)

    if np.issubdtype(dtype, np.unsignedinteger):
        max_value = np.iinfo(dtype).max
    else:
        max_value = np.iinfo(dtype).max

    if vocab_size - 1 > max_value:
        raise ValueError(f"dtype {dtype_name} is too small for vocab_size={vocab_size}")

    return dtype


def encode_txt_as_memmap(tokenizer: Tokenizer, input_path: Path, output_path: Path, dtype: np.dtype) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    num_lines = count_lines(input_path)

    total_tokens = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc=f"Counting {input_path.name}"):
            total_tokens += len(tokenizer.encode(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_ids = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_tokens,))

    position = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc=f"Writing {output_path.name}"):
            ids = tokenizer.encode(line)
            next_position = position + len(ids)
            token_ids[position:next_position] = ids
            position = next_position

    token_ids.flush()
    return total_tokens


def run_sanity_check(tokenizer: Tokenizer, input_path: Path, sanity_chars: int) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        text = f.read(sanity_chars)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("\n=== Sanity check ===")
    print(f"input chars: {len(text)}")
    print(f"token ids: {len(ids)}")
    print(f"decode exactly matches input: {decoded == text}")
    print(f"first 20 ids: {ids[:20]}")


def main() -> None:
    args = parse_args()

    vocab_path = args.tokenizer_dir / "vocab.pkl"
    merges_path = args.tokenizer_dir / "merges.pkl"
    special_tokens = load_special_tokens(args.tokenizer_dir)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=special_tokens,
    )
    dtype = choose_dtype(args.dtype, vocab_size=len(tokenizer.vocab))

    print("Loaded tokenizer")
    print(f"tokenizer_dir: {args.tokenizer_dir}")
    print(f"vocab size: {len(tokenizer.vocab)}")
    print(f"special tokens: {special_tokens}")
    print(f"output dtype: {dtype}")

    run_sanity_check(tokenizer, args.train_txt, args.sanity_chars)

    train_path = args.output_dir / "train.dat"
    valid_path = args.output_dir / "valid.dat"

    train_tokens = encode_txt_as_memmap(tokenizer, args.train_txt, train_path, dtype)
    valid_tokens = encode_txt_as_memmap(tokenizer, args.valid_txt, valid_path, dtype)

    meta = {
        "tokenizer_dir": str(args.tokenizer_dir),
        "train_txt": str(args.train_txt),
        "valid_txt": str(args.valid_txt),
        "train_dat": str(train_path),
        "valid_dat": str(valid_path),
        "dtype": dtype.name,
        "vocab_size": len(tokenizer.vocab),
        "special_tokens": special_tokens,
        "train_tokens": train_tokens,
        "valid_tokens": valid_tokens,
        "created_at": datetime.now().isoformat(),
    }

    meta_path = args.output_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nTokenization complete")
    print(f"train tokens: {train_tokens:,} -> {train_path}")
    print(f"valid tokens: {valid_tokens:,} -> {valid_path}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
