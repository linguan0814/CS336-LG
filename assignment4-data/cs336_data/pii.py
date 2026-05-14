"""PII masking helpers for pretraining data filtering."""

from __future__ import annotations

import re

EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"
PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"
IP_PLACEHOLDER = "|||IP_ADDRESS|||"

EMAIL_RE = re.compile(
    r"(?<![\w|])[\w.!#$%&'*+/=?^`{}~-]+@(?:[\w-]+\.)+[A-Za-z]{2,}(?![\w|])"
)

PHONE_RE = re.compile(
    r"""
    (?<![\w|])
    (?:\+?1[\s.-]?)?
    (?:
        \(\d{3}\)[\s.-]? |
        \(\d{3}\)- |
        \d{3}[\s.-]?
    )
    \d{3}[\s.-]?\d{4}
    (?![\w|])
    """,
    re.VERBOSE,
)

IP_RE = re.compile(r"(?<![\w|])(?:\d{1,3}\.){3}\d{1,3}(?![\w|])")


def mask_emails(text: str) -> tuple[str, int]:
    """Replace email addresses with the assignment email placeholder."""
    return EMAIL_RE.subn(EMAIL_PLACEHOLDER, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace common US phone number formats with the phone placeholder."""
    return PHONE_RE.subn(PHONE_PLACEHOLDER, text)


def _valid_ipv4(candidate: str) -> bool:
    parts = candidate.split(".")
    if len(parts) != 4:
        return False
    return all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def mask_ips(text: str) -> tuple[str, int]:
    """Replace valid IPv4 addresses with the IP placeholder."""
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        candidate = match.group(0)
        if not _valid_ipv4(candidate):
            return candidate
        replacements += 1
        return IP_PLACEHOLDER

    return IP_RE.sub(replace, text), replacements
