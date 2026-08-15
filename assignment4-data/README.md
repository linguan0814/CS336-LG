# Pretraining Data Engineering

[中文](README.zh-CN.md) | English

This module turns noisy web text into a small, auditable pretraining-data pipeline. It implements HTML extraction, language identification, PII masking, Gopher-style quality filtering, exact line deduplication, and JSONL statistics on toy, public-web, and bounded Common Crawl WET samples.

## Pipeline

```text
Raw HTML / WET text
  → extraction → language identification → PII masking
  → quality filtering → exact line deduplication → filtered corpus
```

## Included components

- `cs336_data/`: extraction, language, PII, quality, deduplication, and pipeline modules.
- `scripts/`: toy, public-web, and bounded WET sample entry points.
- `tests/`: unit tests and compact fixtures.
- `docs/`: detailed filtering report, run notes, and limitations.
- `figures/`: lightweight filtering-funnel and retention plots.

## Run

```bash
uv run pytest -q
uv run python scripts/run_filter_pipeline.py \
  --input examples/raw_html_samples.jsonl \
  --output results/filtered_samples.jsonl \
  --stats results/filter_stats.json
```

The recorded toy sample retains 2 of 4 documents. The bounded public-web run retains 8 of 10 pages, and the 100-record WET sample retains 31 documents. These are bounded data-engineering demonstrations, not full Common Crawl processing or production classifiers. See the [English report](docs/data_filtering_report.md) or [中文报告](docs/data_filtering_report.zh-CN.md) for methods and caveats.
