#!/usr/bin/env python3
"""Print a compact summary of filtering statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize filter_stats.json.")
    parser.add_argument("--stats", type=Path, default=Path("results/filter_stats.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stats.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats}")
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
