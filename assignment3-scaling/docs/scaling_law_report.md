# Scaling Law Report: Public IsoFLOPs Fitting

## Problem Definition

The goal of this module is to estimate compute-optimal scaling relationships from public IsoFLOPs curves. For a fixed training compute budget, the data contains several runs with different model sizes and final losses. The best point on each curve is used as an empirical estimate of the compute-optimal model size for that budget.

## Variables

- `N`: number of model parameters.
- `D`: number of training tokens.
- `C`: training compute budget in FLOPs.

For dense Transformer language models, a common approximation for training compute is:

```text
C ~= 6ND
```

The factor of 6 comes from the approximate forward and backward pass cost per parameter-token pair during training. This is an approximation, but it is standard for scaling-law analysis and is sufficient for this public IsoFLOPs fitting exercise.

## IsoFLOPs Method

An IsoFLOPs curve fixes the compute budget `C` and varies model size `N`. Since `C ~= 6ND`, each choice of `N` implies a corresponding token budget:

```text
D = C / (6N)
```

For each compute budget, the script groups all runs with that same `C`, selects the run with the lowest `final_loss`, and records:

- `C`
- `N_opt`, the parameter count of the lowest-loss run
- `final_loss_min`, the minimum final loss at that compute budget
- `D_opt = C / (6 * N_opt)`

The selected points are then fit in log-log space:

```text
N_opt(C) = A_N * C^alpha
D_opt(C) = A_D * C^beta
```

The fitting is performed with `numpy.polyfit(log(C), log(value), 1)`.

## Fitted Results

The generated results are stored in `results/isoflops_fit_results.json`. They include the fitted coefficients, selected IsoFLOPs optimum points, and extrapolated predictions for `C = 1e23` and `C = 1e24` FLOPs.

From the public IsoFLOPs data, the fitted power laws are:

```text
N_opt(C) = 1.163411e+00 * C^0.468683
D_opt(C) = 1.432570e-01 * C^0.531317
```

In log space, the fitted `R^2` values are:

- `N_opt`: 0.978704
- `D_opt`: 0.983351

The generated plots are:

- `figures/n_opt_scaling.png`
- `figures/d_opt_scaling.png`

Each plot shows the selected IsoFLOPs optimum points as scatter points and the fitted power law as a line.

## Predictions

The script writes predictions for:

- `C = 1e23` FLOPs
- `C = 1e24` FLOPs

For each target compute budget, it predicts:

- `N_opt`
- `D_opt`

The current predictions are:

| Compute `C` | Predicted `N_opt` | Predicted `D_opt` |
| --- | ---: | ---: |
| `1e23` | `7.005423e10` | `2.379109e11` |
| `1e24` | `2.061185e11` | `8.085962e11` |

These predictions are extrapolations from the public IsoFLOPs data and should be interpreted as scaling-law estimates, not as results from new training runs.

## Limitations

- This module only reproduces the public IsoFLOPs fitting part of CS336 Assignment 3.
- It does not use the Stanford-only B200 training API.
- It does not submit to the leaderboard.
- It does not train models.
- It does not generate synthetic training results.
- The result is intended to illustrate compute-optimal scaling behavior, not to claim completion of the official large-scale experiment.
