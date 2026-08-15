#!/usr/bin/env python3
"""Generate presentation figures for A4 filtering statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "toy": Path("results/filter_stats.json"),
    "real_web": Path("results/real_filter_stats.json"),
    "cc_wet_100": Path("results/cc_wet_filter_stats_100.json"),
}


def load_stats(path: Path) -> dict[str, int]:
    """Load one filter stats JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing stats file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_funnel(stats: dict[str, int], output: Path, title: str) -> None:
    """Plot document counts through the filtering funnel."""
    labels = ["Input", "Extracted", "Language", "Quality", "Output"]
    values = [
        stats["num_input_documents"],
        stats["num_extracted_documents"],
        stats["num_passed_language_filter"],
        stats["num_passed_gopher_filter"],
        stats["num_output_documents"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#B279A2"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Documents")
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def save_noise_chart(stats_by_run: dict[str, dict[str, int]], output: Path) -> None:
    """Plot PII masking and duplicate-line removals across runs."""
    labels = list(stats_by_run)
    email = [stats_by_run[name]["num_email_masked"] for name in labels]
    phone = [stats_by_run[name]["num_phone_masked"] for name in labels]
    ip = [stats_by_run[name]["num_ip_masked"] for name in labels]
    duplicate = [stats_by_run[name]["num_duplicate_lines_removed"] for name in labels]

    x = range(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    ax.bar([i - 1.5 * width for i in x], email, width, label="Emails", color="#4C78A8")
    ax.bar([i - 0.5 * width for i in x], phone, width, label="Phones", color="#F58518")
    ax.bar([i + 0.5 * width for i in x], ip, width, label="IPs", color="#54A24B")
    ax.bar([i + 1.5 * width for i in x], duplicate, width, label="Duplicate lines", color="#B279A2")

    ax.set_title("PII Masking and Boilerplate Removal")
    ax.set_ylabel("Count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def save_retention_chart(stats_by_run: dict[str, dict[str, int]], output: Path) -> None:
    """Plot final output retention ratio across runs."""
    labels = list(stats_by_run)
    retention = [
        stats_by_run[name]["num_output_documents"] / stats_by_run[name]["num_input_documents"]
        for name in labels
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=180)
    bars = ax.bar(labels, retention, color=["#4C78A8", "#72B7B2", "#F58518"])
    ax.set_title("Final Document Retention")
    ax.set_ylabel("Output / Input")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, retention, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0%}", ha="center", va="bottom")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A4 filtering statistics.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_by_run = {name: load_stats(path) for name, path in RUNS.items()}

    save_funnel(
        stats_by_run["real_web"],
        args.figures_dir / "real_web_filter_funnel.png",
        "Real Web Filtering Funnel",
    )
    save_funnel(
        stats_by_run["cc_wet_100"],
        args.figures_dir / "cc_wet_filter_funnel.png",
        "Common Crawl WET Filtering Funnel",
    )
    save_noise_chart(stats_by_run, args.figures_dir / "pii_and_dedup_counts.png")
    save_retention_chart(stats_by_run, args.figures_dir / "retention_by_run.png")

    print("Generated figures:")
    for path in sorted(args.figures_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
