"""Lightweight harmful-content classifiers for local demos."""

from __future__ import annotations

import re


def _contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def classify_nsfw(text: str) -> tuple[str, float]:
    """Heuristic NSFW detector for tests and demos, not production use."""
    obscene_patterns = [r"c\*ck", r"f\*ck", r"\bcunts?\b", r"\bassh\*le\b"]
    if _contains_any(text, obscene_patterns):
        return "nsfw", 0.95
    return "non-nsfw", 0.55


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Heuristic toxicity detector for tests and demos, not production use."""
    toxic_patterns = [r"\bidiot\b", r"\bmoron\b", r"\bfuck(?:ers?)?\b", r"\btwat\b"]
    if _contains_any(text, toxic_patterns):
        return "toxic", 0.95
    return "non-toxic", 0.55
