from pathlib import Path
from pandas.errors import EmptyDataError
import pandas as pd
import matplotlib.pyplot as plt
import re

EXPORT_DIR = Path("results/cs336A1")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "train/loss",
    "train/avg_loss",
    "val/loss",
    "train/perplexity",
    "val/perplexity",
    "train/learning_rate",
]

X_CANDIDATES = ["iteration", "_step"]


def safe_name(s: str) -> str:
    return s.replace("/", "_")


def clean_label(raw: str) -> str:
    s = str(raw)

    # remove wandb id suffix
    s = re.sub(r"_[a-z0-9]{8,}$", "", s)

    # only keep useful config info
    if "l8-d768-h12-ctx256-bs8-steps30000-lr0.0003" in s:
        return "L8 d768 h12 ctx256 bs8 lr=3e-4"
    if "l2-d128-h4-ctx256-bs2-steps3-lr0.001" in s:
        return "debug L2 d128 steps=3"
    if "smoke-test" in s:
        return "smoke test"

    return s[:60]


def is_main_run(run_dir_name: str) -> bool:
    # 主报告只保留 finished 的正式 30k-step run。
    return (
        "l8-d768-h12-ctx256-bs8-steps30000-lr0.0003" in run_dir_name
        and "eot-im" in run_dir_name
    )


def choose_x(df: pd.DataFrame):
    for c in X_CANDIDATES:
        if c in df.columns:
            return c
    return None


def load_main_histories():
    histories = []

    for hp in EXPORT_DIR.rglob("history.csv"):
        run_dir = hp.parent

        if not is_main_run(run_dir.name):
            continue

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


def main():
    histories = load_main_histories()

    if not histories:
        raise SystemExit("No main A1 history found. Check run directory names.")

    for metric in METRICS:
        valid = [df for df in histories if metric in df.columns]
        if not valid:
            continue

        plt.figure(figsize=(8, 5))
        plotted = False
        x_used = None

        for df in valid:
            x = choose_x(df)
            if x is None:
                continue

            sub = df[[x, metric, "__run"]].dropna()
            if sub.empty:
                continue

            label = sub["__run"].iloc[0]
            plt.plot(sub[x], sub[metric], linewidth=1.8, label=label)
            plotted = True
            x_used = x

        if not plotted:
            plt.close()
            continue

        plt.title(f"A1 Transformer LM: {metric}")
        plt.xlabel(x_used)
        plt.ylabel(metric)
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()

        out = FIG_DIR / f"a1__{safe_name(metric)}.png"
        plt.savefig(out, dpi=180)
        plt.close()
        print("saved:", out)


if __name__ == "__main__":
    main()
