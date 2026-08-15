# Scaling Law 报告：公开 IsoFLOPs 拟合

[中文](scaling_law_report.zh-CN.md) | [English](scaling_law_report.md)

## 问题定义

目标是从公开 IsoFLOPs 曲线估计 compute-optimal scaling 关系。固定训练计算量 `C`，数据包含不同模型规模和最终 loss 的多次运行；每条曲线的最低 loss 点作为该计算预算下的经验最优模型规模。

## 变量与方法

- `N`：模型参数量。
- `D`：训练 token 数。
- `C`：训练计算量，单位 FLOPs。

采用 dense Transformer 的常见近似：`C ≈ 6ND`，因此 `D = C / (6N)`。脚本按 `C` 分组，选择最低 `final_loss` 的 `N_opt`，计算 `D_opt`，再用 `numpy.polyfit(log(C), log(value), 1)` 拟合：

```text
N_opt(C) = A_N * C^alpha
D_opt(C) = A_D * C^beta
```

## 拟合结果

```text
N_opt(C) = 1.163411e+00 * C^0.468683
D_opt(C) = 1.432570e-01 * C^0.531317
```

log 空间拟合的 R² 为：`N_opt=0.978704`，`D_opt=0.983351`。

| 计算量 `C` | 预测 `N_opt` | 预测 `D_opt` |
| --- | ---: | ---: |
| `1e23` | `7.005423e10` | `2.379109e11` |
| `1e24` | `2.061185e11` | `8.085962e11` |

这些预测是基于公开数据的 scaling-law 外推，不是新训练运行。

## 限制

本模块只复现公开 IsoFLOPs 拟合，不使用 Stanford-only B200 API，不训练模型，不提交 leaderboard，也不声称完成官方大规模实验。
