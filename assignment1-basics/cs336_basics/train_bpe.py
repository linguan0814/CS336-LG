import regex as re
from collections import Counter
import os
import heapq
from tqdm import tqdm


class _ReversePair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    show_progress: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    #initialize the vocabulary
    vocab = _init_vocab({}, special_tokens)
    #pretokenization
    cnt_pretokens = Counter()

    # 1MB 分块读取，留出 100 个字符的尾巴防止 UTF-8 截断
    CHUNK_SIZE = 4 * 1024 * 1024 
    TAIL_SIZE = 4096       
    carry = ""

    file_size = os.path.getsize(input_path)
    read_progress = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc="Pretokenizing",
        disable=not show_progress,
    )

    with open(input_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            read_progress.update(len(chunk.encode("utf-8")))

            # 将上一轮剩下的尾巴和这一轮的新内容拼起来
            buffer = carry + chunk

            # 如果内容够长，就切掉最后 TAIL_SIZE 个字符留到下一轮处理
            if len(buffer) > TAIL_SIZE:
                safe_text = buffer[:-TAIL_SIZE]
                carry = buffer[-TAIL_SIZE:]
            else:
                # 如果文件很小或者还没读够 TAIL_SIZE，先全留着
                safe_text = ""
                carry = buffer

            # 只处理安全的文本
            if safe_text:
                for token in pre_tokenization(safe_text, special_tokens):
                    cnt_pretokens[word_2_byte(token)] += 1
    read_progress.close()

    # 循环结束后，处理最后剩下的尾巴
    if carry:
        for token in pre_tokenization(carry, special_tokens):
            cnt_pretokens[word_2_byte(token)] += 1
    #merge
    merge_rule = []
    merge_progress = tqdm(
        total=max(vocab_size - len(vocab), 0),
        desc="Learning merges",
        disable=not show_progress,
    )

    words = {}
    word_counts = {}
    pair_counts = Counter()
    pair_to_words = {}

    for word_id, (pretoken, cnt) in enumerate(cnt_pretokens.items()):
        word = list(pretoken)
        words[word_id] = word
        word_counts[word_id] = cnt
        for pair in _adjacent_pairs(word):
            pair_counts[pair] += cnt
            pair_to_words.setdefault(pair, set()).add(word_id)

    pair_heap = []
    for pair, cnt in pair_counts.items():
        heapq.heappush(pair_heap, (-cnt, _ReversePair(pair), pair))

    while len(vocab) < vocab_size:
        merge_pair = None
        while pair_heap:
            neg_cnt, _, pair = heapq.heappop(pair_heap)
            current_cnt = pair_counts.get(pair, 0)
            if current_cnt > 0 and current_cnt == -neg_cnt:
                merge_pair = pair
                break
        if merge_pair is None:
            break

        merge_rule.append(merge_pair)
        n = len(vocab)
        new_token = merge_pair[0] + merge_pair[1]
        vocab[n] = new_token

        affected_word_ids = list(pair_to_words.get(merge_pair, ()))
        if not affected_word_ids:
            pair_counts[merge_pair] = 0
            continue

        touched_pairs = set()
        for word_id in affected_word_ids:
            word = words[word_id]
            cnt = word_counts[word_id]
            old_pairs = _adjacent_pairs(word)
            if merge_pair not in old_pairs:
                continue

            for pair in old_pairs:
                pair_counts[pair] -= cnt
                touched_pairs.add(pair)
                word_ids = pair_to_words.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_id)

            new_word = _merge_word(word, merge_pair, new_token)
            words[word_id] = new_word

            for pair in _adjacent_pairs(new_word):
                pair_counts[pair] += cnt
                pair_to_words.setdefault(pair, set()).add(word_id)
                touched_pairs.add(pair)

        for pair in touched_pairs:
            cnt = pair_counts.get(pair, 0)
            if cnt > 0:
                heapq.heappush(pair_heap, (-cnt, _ReversePair(pair), pair))

        merge_progress.update(1)

    merge_progress.close()

    return vocab, merge_rule


def pre_tokenization(s: str, special_token: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # ① 没有 special 也要按正则切
    if not special_token:
        for m in re.finditer(PAT, s):
            yield m.group(0)
        return

    # ③ 长→短排序，防止短的抢先匹配
    toks = sorted(special_token, key=len, reverse=True)
    union = "|".join(re.escape(t) for t in toks)
    parts = re.split(f"({union})", s)

    #out = []
    st = set(special_token)
    for part in parts:
        if not part:
            continue
        # ② special 只作为边界，完全跳过
        if part in st:
            continue
        for m in re.finditer(PAT, part): 
            yield m.group(0)
        #out.extend(re.findall(PAT, part))
    #return out

# #multiprocessing's pretoken worker
# def pretoken_worker(input_path, start, end, special_tokens, out_q):
#     import regex as re
#     with open(input_path, "rb") as f:
#         f.seek(start)
#         data = f.read(end - start)
#     text = data.decode("utf-8")  # 严格解码

#     # special 处理（长度降序 + re.escape）
#     toks = sorted(special_tokens, key=len, reverse=True)
#     union = "|".join(re.escape(t) for t in toks)
#     parts = re.split(f"({union})", text)

#     from collections import Counter
#     cnt = Counter()
#     for part in parts:
#         if not part or part in special_tokens:
#             continue
#         for m in re.finditer(PAT, part):
#             token = tuple(word_2_byte(m.group(0)))
#             cnt[token] += 1

#     out_q.put(cnt)

def _init_vocab(vocab: dict, special_token:list):
    special_token_encoded = [s.encode('UTF-8') for s in special_token]
    idx = 0
    for code in special_token_encoded:
        vocab[idx] = code
        idx += 1
    
    for i in range(256):
        init_str = bytes([i])
        if init_str not in vocab.values():
            vocab[idx] = init_str
            idx += 1
    return vocab

def word_2_byte(word: str) -> tuple[bytes, ...]:
    word_decoded = list(word.encode('UTF-8'))
    #split the bytes
    word_byte = [bytes([b]) for b in word_decoded]
    return tuple(word_byte)


def _adjacent_pairs(word: list[bytes]) -> list[tuple[bytes, bytes]]:
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def _merge_word(
    word: list[bytes],
    merge_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> list[bytes]:
    first, second = merge_pair
    new_word = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == first and word[i + 1] == second:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word
