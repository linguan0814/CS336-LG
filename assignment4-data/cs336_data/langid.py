"""Language identification with a deterministic local fallback."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_fasttext_model():
    """Load fastText if explicitly configured.

    The portfolio pipeline defaults to a deterministic fallback because the
    production `lid.176.bin` model is large and is not checked into this repo.
    Set `FASTTEXT_LID_MODEL` to enable fastText locally.
    """
    model_path = os.environ.get("FASTTEXT_LID_MODEL")
    if not model_path:
        return None
    try:
        import fasttext

        return fasttext.load_model(model_path)
    except Exception:
        return None


def _identify_with_fasttext(text: str) -> tuple[str, float] | None:
    model = _load_fasttext_model()
    if model is None:
        return None
    labels, scores = model.predict(text.replace("\n", " "), k=1)
    if not labels:
        return None
    label = labels[0].replace("__label__", "")
    return label, float(scores[0])


def identify_language(text: str) -> tuple[str, float]:
    """Return `(language_code, confidence)` for simple English/Chinese demos.

    Production Common Crawl filtering should use fastText `lid.176.bin`. The
    fallback here is deterministic and dependency-light for local tests.
    """
    if not text or not text.strip():
        return "unknown", 0.0

    fasttext_result = _identify_with_fasttext(text)
    if fasttext_result is not None:
        return fasttext_result

    non_space_chars = [char for char in text if not char.isspace()]
    if not non_space_chars:
        return "unknown", 0.0

    cjk_chars = sum("\u4e00" <= char <= "\u9fff" for char in non_space_chars)
    latin_chars = sum(("a" <= char.lower() <= "z") for char in non_space_chars)
    ascii_chars = sum(char.isascii() for char in non_space_chars)
    total = len(non_space_chars)

    if cjk_chars / total >= 0.30:
        return "zh", 0.99
    if latin_chars >= 20 and ascii_chars / total >= 0.70:
        return "en", 0.99
    if latin_chars / total >= 0.50:
        return "en", 0.85
    return "unknown", 0.0
