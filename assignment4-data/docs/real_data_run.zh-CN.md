# 真实网页样本运行记录

[中文](real_data_run.zh-CN.md) | [English](real_data_run.md)

这是 A4 filtering pipeline 的小规模真实数据验证：从 `data/real_web_urls.txt` 读取少量公开 URL，抓取后执行同一套过滤流程。它不是广泛爬取，也不是 full Common Crawl 处理。

## 运行

```bash
uv run python scripts/fetch_web_pages.py \
  --urls data/real_web_urls.txt \
  --output results/real_raw_html.jsonl \
  --limit 10 --timeout 10 --sleep 0.5

uv run python scripts/run_filter_pipeline.py \
  --input results/real_raw_html.jsonl \
  --output results/real_filtered_samples.jsonl \
  --stats results/real_filter_stats.json

uv run python scripts/summarize_filter_stats.py --stats results/real_filter_stats.json
```

## 观察指标

- 成功抓取的页面数量；
- 英文过滤和 Gopher 质量过滤的拒绝数量；
- PII-like pattern 的遮盖数量；
- 重复 boilerplate 行的删除数量。

当前默认使用本地 deterministic language fallback；生产规模应配置 `FASTTEXT_LID_MODEL` 和 fastText `lid.176.bin`。公开站点可能拒绝脚本请求或返回非 HTML，这属于预期情况。
