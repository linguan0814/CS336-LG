import numpy as np
from tqdm import tqdm
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cs336_basics.tokenizer import Tokenizer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOKENIZER_DIR = PROJECT_ROOT / "artifacts" / "tokenizers" / "tinystories_bpe_vocab10000_eot-im"
VOCAB_PATH = TOKENIZER_DIR / "vocab.pkl"
MERGES_PATH = TOKENIZER_DIR / "merges.pkl"

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_TXT_DATA_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VAL_TXT_DATA_PATH = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "tokenized" / "tinystories_bpe_vocab10000_eot-im"
TRAIN_DATA_PATH = OUTPUT_DIR / "train.dat"
VAL_DATA_PATH = OUTPUT_DIR / "valid.dat"

special_tokens = ["<|endoftext|>"]

def encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path):
    with open(path_to_txt, 'r') as f:
        num_lines = sum(1 for _ in f)
    
    # 第一步：统计总token数（需要遍历一遍）
    total_tokens = 0
    with open(path_to_txt, 'r') as f:
        for line in tqdm(f, total=num_lines, desc="Counting tokens"):
            total_tokens += len(tokenizer.encode(line))

    # 第二步：创建memmap
    dtype = np.int32
    tokens_mm = np.memmap(save_path, dtype=dtype, mode='w+', shape=(total_tokens,))

    # 第三步：再次遍历写入
    pos = 0
    with open(path_to_txt, 'r') as f:
        for line in tqdm(f, total=num_lines, desc="Tokenizing"):
            ids = tokenizer.encode(line)
            n = len(ids)
            tokens_mm[pos:pos+n] = ids
            pos += n

    tokens_mm.flush()

def main():
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(VOCAB_PATH),
        merges_filepath=str(MERGES_PATH),
        special_tokens=special_tokens,
    )

    print("=== 测试 Tokenizer ===")
    test_texts = [
        "Once upon a time, there was a little robot.",
        "Hello world! <|endoftext|> Some more text.",
        "<|endoftext|>",
        "你好，世界！"
    ]

    for text in test_texts:
        print(f"\n原文: {text}")
        encoded = tokenizer.encode(text)
        print("编码:", encoded)

        byte_tokens = [tokenizer.vocab[token_id] for token_id in encoded]
        str_tokens = [b.decode("utf-8", errors="replace") for b in byte_tokens]
        print("分词（可读）:", str_tokens)

        decoded = tokenizer.decode(encoded)
        print("解码:", decoded)
        print("是否完全还原:", decoded == text)

    encode_txt_as_numpy_array(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
    encode_txt_as_numpy_array(tokenizer, VAL_TXT_DATA_PATH, VAL_DATA_PATH)


if __name__ == "__main__":
    main()
