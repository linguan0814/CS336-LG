# Real Web Data Run

This document records the small real-data run for the A4 filtering pipeline. It is intentionally modest: the goal is to demonstrate the pipeline on public web pages without doing broad crawling or large-scale Common Crawl processing.

## Scope

- Input: a small curated URL list in `data/real_web_urls.txt`
- Fetch limit: recommended 10 to 20 pages
- Output raw HTML: `results/real_raw_html.jsonl`
- Output filtered text: `results/real_filtered_samples.jsonl`
- Output stats: `results/real_filter_stats.json`

This run is not full Common Crawl scale and does not train a leaderboard model.

## Commands

From `assignment4-data/`:

```bash
uv run python scripts/fetch_web_pages.py \
  --urls data/real_web_urls.txt \
  --output results/real_raw_html.jsonl \
  --limit 10 \
  --timeout 10 \
  --sleep 0.5
```

Then run the existing filtering pipeline:

```bash
uv run python scripts/run_filter_pipeline.py \
  --input results/real_raw_html.jsonl \
  --output results/real_filtered_samples.jsonl \
  --stats results/real_filter_stats.json
```

Summarize the result:

```bash
uv run python scripts/summarize_filter_stats.py --stats results/real_filter_stats.json
```

Inspect the first cleaned samples:

```bash
head -n 3 results/real_filtered_samples.jsonl
```

## What To Look For

- How many pages were fetched successfully.
- How many pages were rejected by the English language filter.
- How many pages failed Gopher-style quality filtering.
- Whether any synthetic-looking PII patterns were masked.
- Whether repeated boilerplate lines were removed.

## Notes

Some public sites may block scripted requests or return non-HTML content. That is expected. The fetch script logs failures and continues.

The default language ID is the local fallback unless `FASTTEXT_LID_MODEL` is configured. For production-scale filtering, use fastText `lid.176.bin`.
