#!/usr/bin/env python3
"""Fetch a small list of public web pages into raw HTML JSONL.

This script is intentionally conservative: it is for a small real-data demo,
not broad crawling. It reads one URL per line, skips comments, fetches at most
`--limit` pages, and caps bytes per response.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = "CS336-LG-data-filtering-demo/0.1 (+local educational project)"


def load_urls(path: Path) -> list[str]:
    """Load non-empty, non-comment URLs from a text file."""
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {path}")

    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    return urls


def fetch_url(url: str, timeout: float, max_bytes: int) -> tuple[str, str]:
    """Fetch one URL and return `(html, final_url)`."""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise ValueError(f"Skipping non-HTML content type: {content_type}")
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace"), response.geturl()


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write JSONL records with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a small public-web HTML sample.")
    parser.add_argument("--urls", type=Path, required=True, help="Text file with one URL per line.")
    parser.add_argument("--output", type=Path, required=True, help="Output raw HTML JSONL file.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of URLs to attempt.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests in seconds.")
    parser.add_argument("--max-bytes", type=int, default=1_000_000, help="Maximum bytes to read per page.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = load_urls(args.urls)[: args.limit]
    records: list[dict[str, object]] = []
    num_failed = 0

    for index, url in enumerate(urls, start=1):
        try:
            html, final_url = fetch_url(url, timeout=args.timeout, max_bytes=args.max_bytes)
        except (HTTPError, URLError, TimeoutError, ValueError, UnicodeError) as exc:
            num_failed += 1
            print(f"[{index}/{len(urls)}] failed: {url} ({exc})")
            time.sleep(args.sleep)
            continue

        record = {
            "id": f"real_web_{index:04d}",
            "url": final_url,
            "source_url": url,
            "html": html,
        }
        records.append(record)
        print(f"[{index}/{len(urls)}] fetched: {final_url} ({len(html)} chars)")
        time.sleep(args.sleep)

    write_jsonl(args.output, records)
    print("Fetch complete.")
    print(f"attempted_urls: {len(urls)}")
    print(f"successful_pages: {len(records)}")
    print(f"failed_pages: {num_failed}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
