# Limitations

This A4 module is intentionally scoped as a portfolio-ready core data filtering pipeline.

What it does not claim:

- It is not full Common Crawl scale.
- It does not train a leaderboard model.
- It does not download or bundle large external classifiers.
- It does not provide production harmful-content classification.
- It does not provide production learned quality classification.
- It does not implement scalable MinHash/LSH fuzzy deduplication.

Current lightweight fallbacks:

- Language identification uses fastText only if `FASTTEXT_LID_MODEL` is configured; otherwise it uses deterministic English/Chinese heuristics.
- `classify_quality`, `classify_nsfw`, and `classify_toxic_speech` are local deterministic placeholders for tests and demos.
- Near-deduplication uses exact Jaccard over token ngrams for small fixtures, not a scalable MinHash pipeline.

These choices keep the module honest, runnable, and easy to inspect. The production path would replace these fallbacks with trained classifiers and distributed deduplication.
