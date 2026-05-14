"""Lightweight quality filters for web pretraining data."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"\S+")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def gopher_quality_filter(text: str) -> bool:
    """Apply the CS336 core subset of Gopher-style quality rules."""
    if not text or not text.strip():
        return False

    words = _words(text)
    if not 50 <= len(words) <= 100_000:
        return False

    mean_word_length = sum(len(word) for word in words) / len(words)
    if not 3 <= mean_word_length <= 10:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(line.endswith("...") for line in lines)
        if ellipsis_lines / len(lines) > 0.30:
            return False

    alpha_words = sum(any(char.isalpha() for char in word) for word in words)
    if alpha_words / len(words) < 0.80:
        return False

    return True


def classify_quality(text: str) -> tuple[str, float]:
    """Deterministic fixture-scale quality classifier.

    This is not a production classifier. It separates encyclopedia-like text
    from forum/template-heavy common-crawl text for local smoke tests and
    project demos. A production pipeline should use a trained quality model.
    """
    lowered = text.lower()
    cc_markers = [
        "forum index",
        "memberlist",
        "log in",
        "powered by",
        "teach english abroad",
        "contact",
    ]
    wiki_markers = [
        "first published",
        "substantive revision",
        "political theory",
        "references",
        "philosophical",
    ]

    cc_score = sum(marker in lowered for marker in cc_markers)
    wiki_score = sum(marker in lowered for marker in wiki_markers)
    if wiki_score >= cc_score:
        return "wiki", float(max(0.5, wiki_score))
    return "cc", float(max(0.5, cc_score))
