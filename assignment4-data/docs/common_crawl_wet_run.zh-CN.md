# Common Crawl WET 样本运行记录

[中文](common_crawl_wet_run.zh-CN.md) | [English](common_crawl_wet_run.md)

## 已完成的 100 条样本运行

从 `CC-MAIN-2026-17` 的一个 WET 文件流式读取 100 条文本记录，使用 A4 同一套 pipeline 过滤。

| 指标 | 数值 |
| --- | ---: |
| 输入 WET records | 100 |
| 提取文本 records | 100 |
| 通过语言过滤 | 44 |
| 通过 Gopher 过滤 | 31 |
| 遮盖 email | 16 |
| 遮盖电话 | 25 |
| 遮盖 IP | 0 |
| 删除重复行 | 512 |
| 输出文档 | 31 |

样本包含技术/社区页面、产品文档和音乐页面，也暴露了导航、商品页 boilerplate 与重复模板等真实噪声。

## 运行入口

```bash
uv run python scripts/sample_common_crawl_wet.py \
  --crawl-id CC-MAIN-2026-17 \
  --output results/cc_wet_raw_text_500.jsonl \
  --limit 500 --timeout 30 --min-chars 200
```

随后将输出交给 `scripts/run_filter_pipeline.py` 和 `scripts/summarize_filter_stats.py`。该记录仍是 bounded sample，不代表完整 Common Crawl 规模。

## 为什么先用 WET

WET 已经提供抽取后的网页文本，可以先验证语言、PII、质量和去重逻辑，而无需先实现完整的原始 HTTP/WARC parser。
