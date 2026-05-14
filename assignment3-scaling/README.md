# Scaling Laws: Public IsoFLOPs Analysis

## Overview

This module is the scaling-law analysis component of the CS336-LG portfolio project. It reproduces the public IsoFLOPs fitting part of CS336 Assignment 3, using only `data/isoflops_curves.json`.

It is intentionally small: no private APIs, no training infrastructure, no leaderboard submission, and no synthetic training results. The purpose is to show clear understanding of compute-optimal scaling laws in a way that is transparent and reproducible.

This module reproduces the public IsoFLOPs fitting part of CS336 Assignment 3. It does not use the Stanford-only B200 training API or leaderboard.

## Project Structure

```text
assignment3-scaling/
├── README.md
├── data/
│   └── isoflops_curves.json
├── docs/
│   └── scaling_law_report.md
├── figures/
│   ├── d_opt_scaling.png
│   └── n_opt_scaling.png
├── results/
│   └── isoflops_fit_results.json
├── scripts/
│   └── fit_isoflops.py
├── pyproject.toml
└── uv.lock
```

## What It Does

- Loads public IsoFLOPs run data.
- Groups runs by `compute_budget`.
- Selects the lowest-loss run for each compute budget.
- Computes `D_opt = C / (6 * N_opt)`.
- Fitting:
  - `N_opt(C) = A_N * C^alpha`
  - `D_opt(C) = A_D * C^beta`
- Generates log-log scaling plots.
- Saves fitted coefficients, optimum points, and extrapolations to JSON.

## What It Does Not Do

- It does not train models.
- It does not use the Stanford-only B200 training API.
- It does not submit to a leaderboard.
- It does not include the original assignment server, scheduler, database, or API scaffold.
- It does not claim official large-scale experiment completion.

## Usage

From this directory:

```bash
uv run python scripts/fit_isoflops.py --input data/isoflops_curves.json
```

Equivalent explicit form:

```bash
uv run python scripts/fit_isoflops.py \
  --input data/isoflops_curves.json \
  --figures-dir figures \
  --results results/isoflops_fit_results.json
```

If `uv` is unavailable, use any Python environment with `numpy` and `matplotlib` installed:

```bash
python scripts/fit_isoflops.py --input data/isoflops_curves.json
```

## Current Results

Fit from the public IsoFLOPs data:

| Quantity | Value |
| --- | ---: |
| `alpha` | `0.468683` |
| `beta` | `0.531317` |
| `A_N` | `1.163411e+00` |
| `A_D` | `1.432570e-01` |

Extrapolated compute-optimal estimates:

| Compute `C` | Predicted `N_opt` | Predicted `D_opt` |
| --- | ---: | ---: |
| `1e23` | `7.005423e10` | `2.379109e11` |
| `1e24` | `2.061185e11` | `8.085962e11` |

## Outputs

- `figures/n_opt_scaling.png`
- `figures/d_opt_scaling.png`
- `results/isoflops_fit_results.json`
- `docs/scaling_law_report.md`

## Portfolio Role

In the full CS336-LG project, this is a compact theory and analysis module. It supports the broader systems/data/alignment work by showing that the project understands compute-optimal scaling, while keeping the implementation honest and reproducible.
