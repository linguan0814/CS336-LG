# 数据过滤报告

[中文](data_filtering_report.zh-CN.md) | [English](data_filtering_report.md)

## 1. 动机与 pipeline

LLM 预训练数据通常来自嘈杂网页。在成为可信的训练语料前，需要先将 HTML 转成正文、过滤非目标语言、遮盖明显 PII、拒绝低质量页面，并删除重复 boilerplate。

```text
原始 HTML → 正文提取 → 语言识别 → PII masking → 质量过滤 → 去重 → filtered corpus
```

本模块支持 toy JSONL、10 页公开网页和 100 条 bounded Common Crawl WET 样本。

## 2. 主要规则

- 可选 fastText；没有模型时使用确定性的中英文 fallback。
- email、美国常见电话和合法 IPv4 分别替换为 `|||EMAIL_ADDRESS|||`、`|||PHONE_NUMBER|||`、`|||IP_ADDRESS|||`。
- Gopher 风格规则：词数 `[50,100000]`、平均词长 `[3,10]`、省略号行比例不超过 30%、含字母词比例至少 80%。
- corpus-level exact line deduplication，用于去除重复 footer、导航和模板。

## 3. 记录结果

| 数据源 | 输入 | 通过语言 | 通过质量 | 输出 |
| --- | ---: | ---: | ---: | ---: |
| Toy sample | 4 | 3 | 2 | 2 |
| 公开网页 | 10 | 10 | 8 | 8 |
| Common Crawl WET | 100 | 44 | 31 | 31 |

Toy sample 遮盖 2 个 email、2 个电话和 1 个 IP，删除 2 条重复行。公开网页样本遮盖 17 个 email、1 个电话、50 个 IP，删除 1548 条重复行；WET 样本遮盖 16 个 email、25 个电话，删除 512 条重复行。

## 4. 限制

这是 bounded sample pipeline，不是完整 Common Crawl 处理。fallback language ID、quality/harmful-content classifier 都是轻量 deterministic placeholder；exact dedup 也不能替代生产级 MinHash/LSH。生产路径应接入 fastText、训练好的质量/安全分类器和可扩展分片去重。
