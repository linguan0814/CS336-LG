from pathlib import Path
import pandas as pd

EXPORT_DIR = Path("reports/wandb_exports")
OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for p in EXPORT_DIR.rglob("_runs_index.csv"):
    project = p.parent.name
    df = pd.read_csv(p)
    df.insert(0, "project", project)
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)

all_df.to_csv(OUT_DIR / "wandb_runs_all.csv", index=False)

cols = [c for c in [
    "project", "run_name", "state", "created_at", "url",
    "summary/eval/accuracy",
    "summary/eval/reward",
    "summary/eval/answer_reward",
    "summary/eval/format_reward",
    "summary/train/loss",
    "summary/train/policy_loss",
    "summary/train/entropy",
    "config/lr",
    "config/learning_rate",
    "config/loss_type",
    "config/epochs_per_rollout",
    "config/train_batch_size",
] if c in all_df.columns]

summary = all_df[cols]
summary.to_csv(OUT_DIR / "wandb_runs_summary.csv", index=False)

print("saved reports/tables/wandb_runs_all.csv")
print("saved reports/tables/wandb_runs_summary.csv")
print("shape:", all_df.shape)
print("summary columns:", cols)
