# Scaling Laws：公开 IsoFLOPs 分析

[中文](README.zh-CN.md) | [English](README.md)

这是 CS336-LG 的 scaling-law 分析模块，使用 `data/isoflops_curves.json` 复现公开的 IsoFLOPs 拟合部分。它不依赖 Stanford-only B200 training API、不训练模型、不提交 leaderboard，也不伪造大规模实验结果。

## 做了什么

- 读取公开 IsoFLOPs run 数据，并按 `compute_budget` 分组。
- 每个 compute budget 选择最低 `final_loss` 的 run。
- 根据 `D_opt = C / (6 * N_opt)` 计算 compute-optimal token 数。
- 在 log-log 空间拟合：`N_opt(C) = A_N * C^alpha`、`D_opt(C) = A_D * C^beta`。
- 生成 scaling 图，保存系数、最优点和外推结果。

```text
assignment3-scaling/
├── data/isoflops_curves.json
├── docs/scaling_law_report.md
├── figures/
├── results/isoflops_fit_results.json
├── scripts/fit_isoflops.py
├── pyproject.toml
└── uv.lock
```

## 运行

```bash
uv run python scripts/fit_isoflops.py \
  --input data/isoflops_curves.json \
  --figures-dir figures \
  --results results/isoflops_fit_results.json
```

## 当前拟合结果

| 参数 | 数值 |
| --- | ---: |
| `alpha` | `0.468683` |
| `beta` | `0.531317` |
| `A_N` | `1.163411e+00` |
| `A_D` | `1.432570e-01` |

这些是公开 IsoFLOPs 数据的拟合与外推，不是新训练结果。完整方法、R² 和限制见[中文报告](docs/scaling_law_report.zh-CN.md)。
