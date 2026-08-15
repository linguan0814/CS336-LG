# A1 设计文档

[中文](README.zh-CN.md) | [English](README.md)

## 目标

从原始文本到自回归生成，构建一个能够检查每个组件的 causal language-model pipeline。

## 架构

| 阶段 | 代码 | 职责 |
| --- | --- | --- |
| Tokenization | `train_bpe.py`、`tokenizer.py`、`scripts/train_tokenizer.py` | GPT 风格预分词、BPE merge、序列化、encode/decode |
| 数据 | `scripts/tokenize_dataset.py`、`trainer/data_loading.py` | 文本转 token memmap，采样训练 batch |
| 模型 | `model/modules.py`、`model/transformer.py` | Embedding、RMSNorm、SwiGLU、RoPE、causal attention、Transformer LM |
| 优化 | `trainer/AdamW.py`、`trainer/utils.py` | AdamW、学习率、loss、梯度裁剪 |
| 训练/推理 | `train.py`、`generate.py`、`check_pointing.py` | 训练验证、checkpoint、temperature/top-p 生成 |

## 验证与证据

`tests/` 覆盖关键接口和行为。`assets/a1_validation_loss.png` 是公开保留的单张训练曲线。其他曲线、语料、tokenized 数据、checkpoint 和 tracking 文件仅保留在本地，不影响代码检查或单元测试。

## 复现边界

运行 `uv run pytest` 可执行已提交测试。完整训练还需要本地语料和足够算力；公开仓库不声称 reviewer 无需准备外部输入即可重现原始训练运行。
