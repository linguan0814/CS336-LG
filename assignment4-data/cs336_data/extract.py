"""HTML extraction utilities for the data filtering pipeline."""

from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def _decode_html_bytes(html_bytes: bytes) -> str:
    """Decode HTML bytes, preferring UTF-8 and falling back to detected encoding."""
    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes) or "utf-8"
        return html_bytes.decode(encoding, errors="replace")


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """Extract plain text from HTML bytes.

    Empty input returns an empty string. The extractor uses resiliparse's
    production-oriented HTML-to-text implementation after decoding bytes.
    """
    if not html_bytes:
        return ""

    html = _decode_html_bytes(html_bytes)
    if not html.strip():
        return ""

    text = extract_plain_text(html)
    return text if text is not None else ""
