#!/usr/bin/env python3
"""Stream a bounded Common Crawl WET sample into JSONL text records.

This script is intentionally small-scale. It reads one WET file, stops after a
fixed number of conversion records, and writes JSONL records compatible with
`scripts/run_filter_pipeline.py`.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from warcio.archiveiterator import ArchiveIterator


COMMON_CRAWL_BASE = "https://data.commoncrawl.org/"
DEFAULT_CRAWL_ID = "CC-MAIN-2026-17"
USER_AGENT = "CS336-LG-common-crawl-wet-demo/0.1 (+local educational project)"


def _open_url(url: str, timeout: float):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(request, timeout=timeout)


def choose_wet_url(crawl_id: str, timeout: float) -> str:
    """Choose the first WET path from a Common Crawl crawl's wet.paths.gz."""
    paths_url = f"{COMMON_CRAWL_BASE}crawl-data/{crawl_id}/wet.paths.gz"
    with _open_url(paths_url, timeout=timeout) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for raw_line in gz:
                path = raw_line.decode("utf-8", errors="replace").strip()
                if path:
                    return COMMON_CRAWL_BASE + path
    raise ValueError(f"No WET paths found at {paths_url}")


def iter_wet_records(wet_url: str, timeout: float):
    """Yield conversion records from a gzipped WET URL."""
    with _open_url(wet_url, timeout=timeout) as response:
        with gzip.GzipFile(fileobj=response) as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == "conversion":
                    yield record


def sample_wet_to_jsonl(
    wet_url: str,
    output_path: Path,
    limit: int,
    timeout: float,
    min_chars: int,
    max_chars: int,
) -> dict[str, int | str]:
    """Stream WET records and write a bounded JSONL text sample."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempted = 0
    written = 0
    skipped_short = 0
    truncated = 0

    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_wet_records(wet_url, timeout=timeout):
            attempted += 1
            payload = record.content_stream().read()
            text = payload.decode("utf-8", errors="replace").strip()
            if len(text) < min_chars:
                skipped_short += 1
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
                truncated += 1

            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            item = {
                "id": f"cc_wet_{written + 1:06d}",
                "url": url,
                "source_wet_url": wet_url,
                "text": text,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

            if written >= limit:
                break

    return {
        "wet_url": wet_url,
        "attempted_conversion_records": attempted,
        "written_records": written,
        "skipped_short_records": skipped_short,
        "truncated_records": truncated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text records from one Common Crawl WET file.")
    parser.add_argument("--crawl-id", default=DEFAULT_CRAWL_ID, help="Common Crawl crawl id.")
    parser.add_argument("--wet-url", default=None, help="Explicit WET URL. If omitted, use first path from crawl.")
    parser.add_argument("--output", type=Path, required=True, help="Output raw text JSONL path.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum records to write.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--min-chars", type=int, default=200, help="Skip records shorter than this.")
    parser.add_argument("--max-chars", type=int, default=200_000, help="Truncate records longer than this.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        wet_url = args.wet_url or choose_wet_url(args.crawl_id, timeout=args.timeout)
        stats = sample_wet_to_jsonl(
            wet_url=wet_url,
            output_path=args.output,
            limit=args.limit,
            timeout=args.timeout,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"Failed to sample WET data: {exc}") from exc

    print("Common Crawl WET sampling complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
