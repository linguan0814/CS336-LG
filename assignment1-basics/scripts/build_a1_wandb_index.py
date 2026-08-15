from pathlib import Path
import pandas as pd

EXPORT_DIR = Path("results/cs336A1")
OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for index_path in EXPORT_DIR.rglob("_runs_index.csv"):
    project = index_path.parent.name
    df = pd.read_csv(index_path)
    df.insert(0, "project", project)
    rows.append(df)

if not rows:
    raise SystemExit("No _runs_index.csv found under results/cs336A1")

all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv(OUT_DIR / "a1_wandb_runs_all.csv", index=False)

preferred = [
    "project",
    "run_name",
    "state",
    "created_at",
    "url",
    "summary/train/loss",
    "summary/val/loss",
    "summary/train/perplexity",
    "summary/val/perplexity",
    "summary/perplexity",
    "summary/loss",
    "summary/global_step",
    "summary/_step",
    "config/vocab_size",
    "config/context_length",
    "config/d_model",
    "config/d_ff",
    "config/num_layers",
    "config/num_heads",
    "config/batch_size",
    "config/learning_rate",
    "config/lr",
    "config/max_steps",
    "config/dataset",
]

cols = [c for c in preferred if c in all_df.columns]
summary = all_df[cols]
summary.to_csv(OUT_DIR / "a1_wandb_runs_summary.csv", index=False)

print("saved:", OUT_DIR / "a1_wandb_runs_all.csv")
print("saved:", OUT_DIR / "a1_wandb_runs_summary.csv")
print("shape:", all_df.shape)
print("columns:")
for c in all_df.columns:
    print(" -", c)
