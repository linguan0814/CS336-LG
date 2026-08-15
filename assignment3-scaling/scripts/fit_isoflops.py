#!/usr/bin/env python3
"""Fit public IsoFLOPs scaling laws for CS336 Assignment 3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TARGET_COMPUTE_BUDGETS = (1e23, 1e24)
REQUIRED_FIELDS = ("parameters", "compute_budget", "final_loss")


def load_runs(input_path: Path) -> list[dict[str, float]]:
    """Load and validate public IsoFLOPs runs."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        raw_runs: Any = json.load(f)

    if not isinstance(raw_runs, list):
        raise ValueError(f"Expected a JSON list of runs in {input_path}")

    runs: list[dict[str, float]] = []
    for i, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"Run {i} must be a JSON object")

        missing = [field for field in REQUIRED_FIELDS if field not in raw_run]
        if missing:
            raise ValueError(f"Run {i} is missing required fields: {missing}")

        try:
            parameters = float(raw_run["parameters"])
            compute_budget = float(raw_run["compute_budget"])
            final_loss = float(raw_run["final_loss"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Run {i} contains non-numeric values") from exc

        if parameters <= 0:
            raise ValueError(f"Run {i} has non-positive parameters: {parameters}")
        if compute_budget <= 0:
            raise ValueError(f"Run {i} has non-positive compute_budget: {compute_budget}")
        if not math.isfinite(final_loss):
            raise ValueError(f"Run {i} has non-finite final_loss: {final_loss}")

        runs.append(
            {
                "parameters": parameters,
                "compute_budget": compute_budget,
                "final_loss": final_loss,
            }
        )

    if not runs:
        raise ValueError(f"No runs found in {input_path}")

    return runs


def select_isoflops_optima(runs: list[dict[str, float]]) -> list[dict[str, float]]:
    """Choose the lowest-loss run for each compute budget."""
    best_by_compute: dict[float, dict[str, float]] = {}

    for run in runs:
        compute_budget = run["compute_budget"]
        current_best = best_by_compute.get(compute_budget)
        if current_best is None or run["final_loss"] < current_best["final_loss_min"]:
            n_opt = run["parameters"]
            best_by_compute[compute_budget] = {
                "C": compute_budget,
                "N_opt": n_opt,
                "final_loss_min": run["final_loss"],
                "D_opt": compute_budget / (6.0 * n_opt),
            }

    optima = [best_by_compute[c] for c in sorted(best_by_compute)]
    if len(optima) < 2:
        raise ValueError("Need at least two compute budgets to fit a power law")

    return optima


def fit_power_law(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    """Fit y = A * x^exponent in log-log space."""
    if np.any(x_values <= 0) or np.any(y_values <= 0):
        raise ValueError("Power-law fitting requires positive x and y values")

    log_x = np.log(x_values)
    log_y = np.log(y_values)
    exponent, log_a = np.polyfit(log_x, log_y, 1)
    fitted_log_y = exponent * log_x + log_a
    residuals = log_y - fitted_log_y
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "A": float(np.exp(log_a)),
        "exponent": float(exponent),
        "log_A": float(log_a),
        "r_squared_log_space": r_squared,
    }


def predict(coefficient: float, exponent: float, compute_budget: float) -> float:
    """Predict the optimum value at a target compute budget."""
    if compute_budget <= 0:
        raise ValueError(f"compute_budget must be positive, got {compute_budget}")
    return float(coefficient * compute_budget**exponent)


def plot_scaling(
    compute_budgets: np.ndarray,
    observed_values: np.ndarray,
    coefficient: float,
    exponent: float,
    output_path: Path,
    y_label: str,
    title: str,
    formula_name: str,
) -> None:
    """Create a log-log scatter plot plus fitted power-law curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c_min = float(compute_budgets.min())
    c_max = float(max(compute_budgets.max(), max(TARGET_COMPUTE_BUDGETS)))
    line_x = np.logspace(np.log10(c_min), np.log10(c_max), 256)
    line_y = coefficient * line_x**exponent

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=180)
    ax.scatter(
        compute_budgets,
        observed_values,
        color="#1f77b4",
        label="IsoFLOPs optima",
        zorder=3,
    )
    ax.plot(
        line_x,
        line_y,
        color="#d62728",
        linewidth=2,
        label="Fitted power law",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C (training FLOPs)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()

    annotation = (
        f"{formula_name}(C) = {coefficient:.3e} * C^{exponent:.4f}\n"
        f"exponent = {exponent:.4f}"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_results(
    output_path: Path,
    n_fit: dict[str, float],
    d_fit: dict[str, float],
    optima: list[dict[str, float]],
    predictions: list[dict[str, float]],
) -> None:
    """Write fitting outputs to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "fit_coefficients": {
            "N_opt": {
                "A_N": n_fit["A"],
                "alpha": n_fit["exponent"],
                "log_A_N": n_fit["log_A"],
                "r_squared_log_space": n_fit["r_squared_log_space"],
            },
            "D_opt": {
                "A_D": d_fit["A"],
                "beta": d_fit["exponent"],
                "log_A_D": d_fit["log_A"],
                "r_squared_log_space": d_fit["r_squared_log_space"],
            },
        },
        "optimum_points": optima,
        "predictions": predictions,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit public IsoFLOPs scaling laws from final-loss curves."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/isoflops_curves.json"),
        help="Path to the public IsoFLOPs JSON file.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated scaling plots.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/isoflops_fit_results.json"),
        help="Path for the JSON fitting results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs = load_runs(args.input)
    optima = select_isoflops_optima(runs)

    compute_budgets = np.array([point["C"] for point in optima], dtype=float)
    n_opt = np.array([point["N_opt"] for point in optima], dtype=float)
    d_opt = np.array([point["D_opt"] for point in optima], dtype=float)

    n_fit = fit_power_law(compute_budgets, n_opt)
    d_fit = fit_power_law(compute_budgets, d_opt)

    predictions = []
    for compute_budget in TARGET_COMPUTE_BUDGETS:
        n_prediction = predict(n_fit["A"], n_fit["exponent"], compute_budget)
        d_prediction = predict(d_fit["A"], d_fit["exponent"], compute_budget)
        predictions.append(
            {
                "C": compute_budget,
                "N_opt": n_prediction,
                "D_opt": d_prediction,
            }
        )

    plot_scaling(
        compute_budgets=compute_budgets,
        observed_values=n_opt,
        coefficient=n_fit["A"],
        exponent=n_fit["exponent"],
        output_path=args.figures_dir / "n_opt_scaling.png",
        y_label="Optimal parameter count N_opt",
        title="IsoFLOPs Model-Size Scaling",
        formula_name="N_opt",
    )
    plot_scaling(
        compute_budgets=compute_budgets,
        observed_values=d_opt,
        coefficient=d_fit["A"],
        exponent=d_fit["exponent"],
        output_path=args.figures_dir / "d_opt_scaling.png",
        y_label="Optimal token count D_opt",
        title="IsoFLOPs Data-Size Scaling",
        formula_name="D_opt",
    )
    save_results(args.results, n_fit, d_fit, optima, predictions)

    print("IsoFLOPs fitting complete.")
    print(f"Loaded runs: {len(runs)}")
    print(f"Compute budgets: {len(optima)}")
    print(f"alpha: {n_fit['exponent']:.6f}")
    print(f"beta: {d_fit['exponent']:.6f}")
    print(f"A_N: {n_fit['A']:.6e}")
    print(f"A_D: {d_fit['A']:.6e}")
    for item in predictions:
        print(
            "Prediction "
            f"C={item['C']:.1e}: "
            f"N_opt={item['N_opt']:.6e}, "
            f"D_opt={item['D_opt']:.6e}"
        )
    print(f"Saved results: {args.results}")
    print(f"Saved figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
