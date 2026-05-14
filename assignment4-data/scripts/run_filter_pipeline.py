#!/usr/bin/env python3
"""Run the sample A4 data filtering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_data.pipeline import run_filter_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter raw HTML JSONL samples.")
    parser.add_argument("--input", type=Path, required=True, help="Input raw HTML JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output filtered JSONL file.")
    parser.add_argument("--stats", type=Path, required=True, help="Output filter statistics JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_filter_pipeline(args.input, args.output, args.stats)
    print("Filtering complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
