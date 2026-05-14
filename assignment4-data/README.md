# CS336 Assignment 4: LLM Data Filtering Pipeline

## Overview

This module implements an internship-ready core version of the CS336 Assignment 4 data pipeline. Its goal is to turn noisy web data into a small, auditable filtered corpus suitable for LLM pretraining experiments.

The emphasis is on clear, runnable, explainable data engineering rather than full Common Crawl scale or leaderboard training.

## Implemented Components

- HTML-to-text extraction with `resiliparse`
- Language identification with an optional fastText hook and deterministic English/Chinese fallback
- PII masking for emails, US-style phone numbers, and valid IPv4 addresses
- Gopher-style quality filtering
- Exact line deduplication
- Small JSONL sample filtering pipeline
- Filtering statistics and documentation

## Pipeline

```text
Raw HTML
  -> HTML-to-text extraction
  -> language identification
  -> PII masking
  -> Gopher quality filtering
  -> exact line deduplication
  -> filtered corpus
```

## Usage

Core tests:

```bash
uv run pytest tests/test_extract.py -q
uv run pytest tests/test_pii.py -q
uv run pytest tests/test_quality.py -k gopher -q
uv run pytest tests/test_langid.py -q
uv run pytest tests/test_deduplication.py -k exact -q
```

Run the sample pipeline:

```bash
uv run python scripts/run_filter_pipeline.py \
  --input examples/raw_html_samples.jsonl \
  --output results/filtered_samples.jsonl \
  --stats results/filter_stats.json
```

Summarize stats:

```bash
uv run python scripts/summarize_filter_stats.py --stats results/filter_stats.json
```

Optional small real-web run:

```bash
uv run python scripts/fetch_web_pages.py \
  --urls data/real_web_urls.txt \
  --output results/real_raw_html.jsonl \
  --limit 10 \
  --timeout 10 \
  --sleep 0.5

uv run python scripts/run_filter_pipeline.py \
  --input results/real_raw_html.jsonl \
  --output results/real_filtered_samples.jsonl \
  --stats results/real_filter_stats.json
```

See `docs/real_data_run.md` for the step-by-step real-data demo.

Optional bounded Common Crawl WET sample:

```bash
uv run python scripts/sample_common_crawl_wet.py \
  --crawl-id CC-MAIN-2026-17 \
  --output results/cc_wet_raw_text_500.jsonl \
  --limit 500 \
  --timeout 30 \
  --min-chars 200

uv run python scripts/run_filter_pipeline.py \
  --input results/cc_wet_raw_text_500.jsonl \
  --output results/cc_wet_filtered_500.jsonl \
  --stats results/cc_wet_filter_stats_500.json
```

See `docs/common_crawl_wet_run.md` for the staged Common Crawl sample workflow.

## Results

Current sample run:

| Metric | Value |
| --- | ---: |
| input documents | 4 |
| extracted documents | 4 |
| passed language filter | 3 |
| passed Gopher filter | 2 |
| emails masked | 2 |
| phone numbers masked | 2 |
| IPs masked | 1 |
| duplicate lines removed | 2 |
| output documents | 2 |

Generated outputs:

- `results/filtered_samples.jsonl`
- `results/filter_stats.json`
- `examples/cleaned_samples.jsonl`

Real public-web run:

| Metric | Value |
| --- | ---: |
| input pages | 10 |
| extracted pages | 10 |
| passed language filter | 10 |
| passed Gopher filter | 8 |
| emails masked | 17 |
| phone numbers masked | 1 |
| IPs masked | 50 |
| duplicate lines removed | 1548 |
| output documents | 8 |

Bounded Common Crawl WET run:

| Metric | Value |
| --- | ---: |
| input WET records | 100 |
| extracted text records | 100 |
| passed language filter | 44 |
| passed Gopher filter | 31 |
| emails masked | 16 |
| phone numbers masked | 25 |
| IPs masked | 0 |
| duplicate lines removed | 512 |
| output documents | 31 |

Additional outputs:

- `results/real_raw_html.jsonl`
- `results/real_filtered_samples.jsonl`
- `results/real_filter_stats.json`
- `results/cc_wet_raw_text_100.jsonl`
- `results/cc_wet_filtered_100.jsonl`
- `results/cc_wet_filter_stats_100.json`

Figures:

- `figures/real_web_filter_funnel.png`
- `figures/cc_wet_filter_funnel.png`
- `figures/pii_and_dedup_counts.png`
- `figures/retention_by_run.png`

Generate them with:

```bash
uv run python scripts/plot_filter_stats.py --figures-dir figures
```

## Limitations

- This is a local sample-level pipeline, not full Common Crawl scale processing.
- The default language ID fallback is deterministic and lightweight. Production filtering should use fastText `lid.176.bin`.
- The quality classifier and harmful-content classifiers are lightweight placeholders for local tests and demos, not trained production classifiers.
- Exact line deduplication removes repeated lines but does not catch near-duplicate documents.
- The lightweight near-dedup function is suitable for fixtures only; production MinHash/LSH fuzzy deduplication is future work.
- No leaderboard model training is performed here.

## Connection to CS336-LG

A4 is the pretraining data engineering module of CS336-LG. It connects A3's compute-optimal token budget analysis to the practical question of which tokens should enter a training corpus, and it prepares the project narrative for A5's post-training data and alignment work.
