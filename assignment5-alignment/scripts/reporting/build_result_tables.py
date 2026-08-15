from pathlib import Path
import pandas as pd

EXPORT_DIR = Path("reports/wandb_exports")
OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for index_path in EXPORT_DIR.rglob("_runs_index.csv"):
    project = index_path.parent.name
    df = pd.read_csv(index_path)
    df.insert(0, "project", project)
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv(OUT_DIR / "wandb_runs_all.csv", index=False)

finished = all_df[all_df["state"] == "finished"].copy()

# 不把 A1 的 CS336 smoke/debug 主动放进主表
main = finished[finished["project"].str.lower().str.contains("cs336")]
main = main[main["project"] != "CS336"]

preferred_cols = [
    "project",
    "run_name",
    "state",
    "created_at",
    "summary/eval/accuracy",
    "summary/eval/answer_score",
    "summary/eval/format_score",
    "summary/eval/avg_length",
    "summary/train/loss",
    "summary/train/response_entropy",
    "summary/train/clip_fraction",
    "summary/train/ratio_mean",
    "summary/train/grad_norm",
    "summary/rollout/mean_reward",
    "summary/rollout/max_reward",
    "summary/ei/success_rate",
    "summary/ei/success_question_rate",
    "url",
]

cols = [c for c in preferred_cols if c in main.columns]
main[cols].to_csv(OUT_DIR / "wandb_runs_main_summary.csv", index=False)

print("saved:", OUT_DIR / "wandb_runs_all.csv")
print("saved:", OUT_DIR / "wandb_runs_main_summary.csv")
print("main runs:", len(main))
print(main[["project", "run_name", "state"]].to_string(index=False))
