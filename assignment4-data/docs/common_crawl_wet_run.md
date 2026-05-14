# Common Crawl WET Sample Run

This run is the next step after the small public-web demo. It uses a bounded Common Crawl WET sample: one WET file, streamed from Common Crawl, capped at a small number of text records.

The goal is to demonstrate the same filtering pipeline on real web-corpus data without attempting full Common Crawl scale.

## Completed 100-Record Run

The first bounded WET run sampled 100 text records from `CC-MAIN-2026-17` and filtered them with the same A4 pipeline.

Commands used:

```bash
uv run python scripts/sample_common_crawl_wet.py \
  --crawl-id CC-MAIN-2026-17 \
  --output results/cc_wet_raw_text_100.jsonl \
  --limit 100 \
  --timeout 30 \
  --min-chars 200

uv run python scripts/run_filter_pipeline.py \
  --input results/cc_wet_raw_text_100.jsonl \
  --output results/cc_wet_filtered_100.jsonl \
  --stats results/cc_wet_filter_stats_100.json
```

Results:

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

Example retained records included technical/community pages such as Drupal event pages, product documentation-like pages, and music/composition pages. The sample also showed realistic Common Crawl noise: navigation text, ecommerce boilerplate, and repeated template lines.

## Recommended Larger Follow-Up

From `assignment4-data/`:

```bash
uv run python scripts/sample_common_crawl_wet.py \
  --crawl-id CC-MAIN-2026-17 \
  --output results/cc_wet_raw_text_500.jsonl \
  --limit 500 \
  --timeout 30 \
  --min-chars 200
```

Then filter it:

```bash
uv run python scripts/run_filter_pipeline.py \
  --input results/cc_wet_raw_text_500.jsonl \
  --output results/cc_wet_filtered_500.jsonl \
  --stats results/cc_wet_filter_stats_500.json
```

Summarize:

```bash
uv run python scripts/summarize_filter_stats.py --stats results/cc_wet_filter_stats_500.json
```

Preview:

```bash
head -n 2 results/cc_wet_filtered_500.jsonl
```

## Why WET First

WET files contain Common Crawl extracted text records. They are a better next step than WARC for this project stage because they let us focus on language filtering, PII masking, quality filtering, and deduplication without first building a full raw HTTP response parser.

## Scope Boundary

This is still not full Common Crawl processing. It is a bounded sample used to produce credible, auditable filtering statistics for the CS336-LG data module.
