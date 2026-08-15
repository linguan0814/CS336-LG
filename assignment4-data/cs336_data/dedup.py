"""Exact and lightweight near-duplicate document deduplication."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def exact_line_deduplication(input_files: list[Path], output_directory: Path) -> None:
    """Remove corpus-level repeated lines and write one output per input file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in input_files]
    documents = {path: _read_text(path) for path in paths}

    line_counts: Counter[str] = Counter()
    for document in documents.values():
        line_counts.update(document.splitlines(keepends=True))

    for path, document in documents.items():
        kept_lines = [line for line in document.splitlines(keepends=True) if line_counts[line] == 1]
        (output_directory / path.name).write_text("".join(kept_lines), encoding="utf-8")


def _token_ngrams(text: str, ngrams: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return set()
    if len(tokens) < ngrams:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + ngrams]) for i in range(len(tokens) - ngrams + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def minhash_deduplication(
    input_files: list[Path],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Path,
) -> None:
    """Small-corpus near-deduplication using exact Jaccard over token ngrams.

    This deliberately avoids a full MinHash/LSH implementation. It is suitable
    for local fixtures and demos; production-scale fuzzy dedup is future work.
    """
    del num_hashes, num_bands
    output_directory.mkdir(parents=True, exist_ok=True)

    kept: list[tuple[Path, str, set[tuple[str, ...]]]] = []
    for path_like in input_files:
        path = Path(path_like)
        text = _read_text(path)
        shingles = _token_ngrams(text, ngrams)
        duplicate = any(_jaccard(shingles, kept_shingles) >= jaccard_threshold for _, _, kept_shingles in kept)
        if not duplicate:
            kept.append((path, text, shingles))

    for path, text, _ in kept:
        (output_directory / path.name).write_text(text, encoding="utf-8")
