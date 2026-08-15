"""Small-sample LLM pretraining data filtering pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.langid import identify_language
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import gopher_quality_filter


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object")
            records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _remove_duplicate_lines(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    line_counts: Counter[str] = Counter()
    split_records: list[tuple[dict[str, Any], list[str]]] = []

    for record in records:
        lines = record["text"].splitlines(keepends=True)
        split_records.append((record, lines))
        line_counts.update(line for line in lines if line.strip())

    output: list[dict[str, Any]] = []
    removed = 0
    for record, lines in split_records:
        kept_lines = []
        for line in lines:
            if line.strip() and line_counts[line] > 1:
                removed += 1
                continue
            kept_lines.append(line)
        updated = dict(record)
        updated["text"] = "".join(kept_lines).strip()
        output.append(updated)
    return output, removed


def filter_records(records: list[dict[str, Any]], language: str = "en") -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter raw HTML or raw text records into cleaned pretraining records."""
    stats = {
        "num_input_documents": len(records),
        "num_extracted_documents": 0,
        "num_passed_language_filter": 0,
        "num_passed_gopher_filter": 0,
        "num_email_masked": 0,
        "num_phone_masked": 0,
        "num_ip_masked": 0,
        "num_duplicate_lines_removed": 0,
        "num_output_documents": 0,
    }
    cleaned: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        raw_text = record.get("text")
        html = record.get("html")
        if raw_text is not None:
            if not isinstance(raw_text, str):
                raise ValueError(f"Record {index} field 'text' must be a string")
            text = raw_text
        else:
            if html is None:
                raise ValueError(f"Record {index} must contain either 'html' or 'text'")
            if not isinstance(html, str):
                raise ValueError(f"Record {index} field 'html' must be a string")
            text = extract_text_from_html_bytes(html.encode("utf-8"))

        if not text.strip():
            continue
        stats["num_extracted_documents"] += 1

        detected_language, language_score = identify_language(text)
        if detected_language != language:
            continue
        stats["num_passed_language_filter"] += 1

        text, email_count = mask_emails(text)
        text, phone_count = mask_phone_numbers(text)
        text, ip_count = mask_ips(text)
        stats["num_email_masked"] += email_count
        stats["num_phone_masked"] += phone_count
        stats["num_ip_masked"] += ip_count

        passes_gopher = gopher_quality_filter(text)
        if not passes_gopher:
            continue
        stats["num_passed_gopher_filter"] += 1

        cleaned.append(
            {
                "id": record.get("id", f"record_{index:06d}"),
                "url": record.get("url", ""),
                "language": detected_language,
                "language_score": language_score,
                "pii_counts": {
                    "email": email_count,
                    "phone": phone_count,
                    "ip": ip_count,
                },
                "passes_gopher": passes_gopher,
                "text": text.strip(),
            }
        )

    cleaned, duplicate_lines_removed = _remove_duplicate_lines(cleaned)
    stats["num_duplicate_lines_removed"] = duplicate_lines_removed
    stats["num_output_documents"] = len(cleaned)
    return cleaned, stats


def run_filter_pipeline(input_path: Path, output_path: Path, stats_path: Path) -> dict[str, int]:
    """Run the sample filtering pipeline and write JSONL outputs plus stats."""
    records = _read_jsonl(input_path)
    cleaned, stats = filter_records(records)
    _write_jsonl(output_path, cleaned)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    return stats
