# Data Filtering Report

## 1. Motivation

LLM pretraining data starts as noisy web data. Before it can become a credible training corpus, raw HTML should be converted into text, non-target-language documents should be filtered, obvious PII should be masked, low-quality pages should be rejected, and repeated boilerplate should be removed.

This module implements a compact, reproducible version of that pipeline for CS336-LG.

## 2. Pipeline Overview

```text
Raw HTML -> extraction -> langid -> PII masking -> quality filtering -> dedup -> filtered corpus
```

The local sample pipeline reads `examples/raw_html_samples.jsonl` and writes:

- `results/filtered_samples.jsonl`
- `results/filter_stats.json`

Two real-data runs were also completed:

- a 10-page public web run from `data/real_web_urls.txt`
- a 100-record bounded Common Crawl WET run from `CC-MAIN-2026-17`

## 3. HTML Extraction

HTML bytes are decoded with UTF-8 first. If decoding fails, the extractor falls back to `resiliparse.parse.encoding.detect_encoding`. Plain text is extracted with `resiliparse.extract.html2text.extract_plain_text`.

This keeps extraction simple while relying on a library designed for noisy web HTML.

## 4. Language Identification

The module supports an optional fastText model through the `FASTTEXT_LID_MODEL` environment variable. If no model is configured, it uses a deterministic fallback:

- high CJK character ratio -> `zh`
- mostly ASCII/Latin text -> `en`
- otherwise -> `unknown`

This fallback is useful for local tests and small demos. Production Common Crawl filtering should use fastText `lid.176.bin`.

## 5. PII Masking

The pipeline masks:

- emails as `|||EMAIL_ADDRESS|||`
- US-style phone numbers as `|||PHONE_NUMBER|||`
- valid IPv4 addresses as `|||IP_ADDRESS|||`

IPv4 masking validates that each segment is between 0 and 255.

## 6. Gopher Quality Filtering

The implemented Gopher-style core rules are:

- document word count must be in `[50, 100000]`
- mean word length must be in `[3, 10]`
- no more than 30% of non-empty lines may end in `...`
- at least 80% of words must contain an alphabetic character

These rules catch short snippets, symbol-heavy pages, and obvious low-quality pages without requiring a trained classifier.

## 7. Exact Deduplication

Exact line deduplication counts every line across the corpus and removes lines whose corpus-level count is greater than one. In the sample pipeline this removes repeated footer boilerplate after quality filtering.

The standalone exact dedup function also supports file-based inputs and output directories for the CS336 tests.

## 8. Quantitative Statistics

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

Generated figures:

- `figures/real_web_filter_funnel.png`
- `figures/cc_wet_filter_funnel.png`
- `figures/pii_and_dedup_counts.png`
- `figures/retention_by_run.png`

## 9. Qualitative Before/After Examples

Before:

```text
Contact the demo maintainer at demo@example.com or +1 123-456-7890.
The staging server used the documentation IP 192.0.2.146.
Shared site footer for the CS336-LG sample corpus.
```

After:

```text
Contact the demo maintainer at |||EMAIL_ADDRESS||| or |||PHONE_NUMBER|||.
The staging server used the documentation IP |||IP_ADDRESS|||.
```

The repeated footer is removed by exact line deduplication.

Common Crawl retained sample:

```text
Collage no.1 (Hungarian) is the result of an interest in appropriation,
sampling, and the varied audio plundering prevalent in our culture.
```

This sample is representative of the WET run: real web text is useful, but it often arrives with navigation, templates, storefront language, or event-page boilerplate.

## 10. Error Analysis

The current sample demonstrates three common filtering outcomes:

- English article-like pages pass the pipeline.
- Chinese text is filtered out by the English language filter.
- Very short, ellipsis-heavy marketing text fails Gopher quality filtering.
- The Common Crawl WET sample has a sharper funnel: 44 of 100 sampled records pass English language identification, and 31 pass both language and Gopher quality filters.
- Deduplication removes repeated lines but cannot remove all single-page navigation or template text.

The deterministic fallback language ID is intentionally conservative and should not be treated as a multilingual production classifier.

## 11. Limitations

- This is a local sample-level pipeline, not full Common Crawl processing.
- No leaderboard training is performed.
- The quality classifier and harmful-content classifiers are lightweight deterministic placeholders.
- Exact line deduplication cannot remove paraphrased or near-duplicate documents.
- The fixture-scale near-dedup function is not a production MinHash/LSH implementation.

## 12. Future Work

- Integrate fastText `lid.176.bin` for production language identification.
- Add a trained quality classifier for web text.
- Add a trained harmful-content classifier.
- Implement scalable MinHash/LSH fuzzy deduplication.
- Stream WARC/WET files at larger scale with sharded statistics.
