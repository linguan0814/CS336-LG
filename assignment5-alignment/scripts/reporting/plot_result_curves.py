from pathlib import Path
from pandas.errors import EmptyDataError
import pandas as pd
import matplotlib.pyplot as plt
import re

EXPORT_DIR = Path("reports/wandb_exports")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_GROUPS = {
    "sft": ["cs336-sft"],
    "ei_ablation": ["cs336-ei-ablation"],
    "grpo_lr_sweep": ["cs336-grpo-math12k-after-base-lr-grpo_clip"],
    "grpo_loss_ablation": ["cs336-grpo-math12k-after-base-4loss"],
    "grpo_std_ablation": ["cs336-grpo-math12k-after-base-std"],
    "grpo_length_norm": ["cs336-grpo-math12k-length-norm"],
}

METRICS = [
    "eval/accuracy",
    "eval/answer_score",
    "eval/format_score",
    "eval/avg_length",
    "train/loss",
    "train/response_entropy",
    "train/clip_fraction",
    "train/ratio_mean",
    "train/grad_norm",
    "rollout/mean_reward",
    "rollout/max_reward",
    "ei/success_rate",
    "ei/success_question_rate",
]

X_CANDIDATES = [
    "eval_step",
    "train_step",
    "rollout_step",
    "ei_step",
    "global_step",
    "_step",
]


def safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def choose_x(df: pd.DataFrame, metric: str):
    if metric.startswith("eval/") and "eval_step" in df.columns:
        return "eval_step"
    if metric.startswith("train/") and "train_step" in df.columns:
        return "train_step"
    if metric.startswith("rollout/") and "rollout_step" in df.columns:
        return "rollout_step"
    if metric.startswith("ei/") and "ei_step" in df.columns:
        return "ei_step"

    for c in X_CANDIDATES:
        if c in df.columns:
            return c
    return None


def clean_label(raw: str) -> str:
    """
    Convert W&B export directory names like:
      grpo_lr3e-6_z5pb4ct9
      E3_TB256_LR3e-5_grpo_no_clip_zjeo2bkp
    into readable legend labels:
      lr=3e-6
      E3 TB256 LR3e-5 no_clip
    """
    s = str(raw)

    # Remove trailing W&B run id suffix.
    # Most W&B ids are short lowercase/digit strings after the final underscore.
    s = re.sub(r"_[a-z0-9]{8,}$", "", s)

    # Simplify common prefixes.
    s = s.replace("qwen2.5-math-1.5b-", "")
    s = s.replace("E3_TB256_LR3e-5_", "")
    s = s.replace("grpo_", "")
    s = s.replace("mask_normalize_", "")
    s = s.replace("math12k_", "")

    # Make lr labels compact.
    s = re.sub(r"lr([0-9.eE+-]+)", r"lr=\1", s)

    # Human-readable separators.
    s = s.replace("_", " ")
    s = s.replace("-", " ")

    # Specific cleanup.
    s = s.replace("no clip", "no-clip")
    s = s.replace("grpo baseline", "baseline")
    s = s.replace("with std", "with-std")
    s = s.replace("no std", "no-std")
    s = s.replace("len norm", "length-norm")
    s = s.replace("dapo", "dapo-style")

    return s.strip()


def load_histories(project: str):
    project_dir = EXPORT_DIR / project
    if not project_dir.exists():
        return []

    histories = []

    for run_dir in project_dir.iterdir():
        if not run_dir.is_dir():
            continue

        hp = run_dir / "history.csv"
        if not hp.exists() or hp.stat().st_size == 0:
            continue

        try:
            df = pd.read_csv(hp)
        except EmptyDataError:
            continue

        if df.empty:
            continue

        df["__run"] = clean_label(run_dir.name)
        histories.append(df)

    return histories


def plot_group(group_name: str, projects: list[str]):
    histories = []
    for project in projects:
        for df in load_histories(project):
            df["__project"] = project
            histories.append(df)

    if not histories:
        print("no histories:", group_name)
        return

    for metric in METRICS:
        valid = [df for df in histories if metric in df.columns]
        if not valid:
            continue

        plt.figure(figsize=(9, 5))
        plotted = False
        x_used = None

        for df in valid:
            x = choose_x(df, metric)
            if x is None or x not in df.columns:
                continue

            sub = df[[x, metric, "__run"]].dropna()
            if sub.empty:
                continue

            label = str(sub["__run"].iloc[0])
            plt.plot(
                sub[x],
                sub[metric],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=label,
            )
            plotted = True
            x_used = x

        if not plotted:
            plt.close()
            continue

        plt.title(f"{group_name}: {metric}")
        plt.xlabel(x_used)
        plt.ylabel(metric)
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()

        out = FIG_DIR / f"{safe_name(group_name)}__{safe_name(metric)}.png"
        plt.savefig(out, dpi=180)
        plt.close()
        print("saved:", out)


def main():
    for group_name, projects in PROJECT_GROUPS.items():
        plot_group(group_name, projects)


if __name__ == "__main__":
    main()
