# 限制说明

[中文](limitations.zh-CN.md) | [English](limitations.md)

该 A4 模块定位为适合公开展示的核心数据过滤 pipeline，不声称：

- 完成 full Common Crawl scale 处理；
- 训练 leaderboard 模型；
- 打包大型外部分类器；
- 提供生产级 harmful-content 或 learned quality classification；
- 实现可扩展的 MinHash/LSH fuzzy deduplication。

当前 fallback 包括：没有配置 `FASTTEXT_LID_MODEL` 时使用中英文 heuristic；`classify_quality`、`classify_nsfw`、`classify_toxic_speech` 是本地测试 placeholder；小型 fixture 的近似去重使用 exact Jaccard，而不是生产级 MinHash pipeline。

这些取舍保持模块诚实、可运行、易检查。生产路径应替换为训练好的分类器和分布式去重系统。
