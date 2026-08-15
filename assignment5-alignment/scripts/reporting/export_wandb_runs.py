import json
import os
import re
from pathlib import Path

import pandas as pd
import wandb


# Set WANDB_ENTITY locally before running this exporter.  The public repository
# intentionally does not embed a personal W&B account or project namespace.
ENTITY = os.environ.get("WANDB_ENTITY")

OUT_DIR = Path("reports/wandb_exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", s)
    return s.strip("_") or "unknown"


def export_run(run, project_dir: Path):
    run_name = safe_name(run.name or run.id)
    run_dir = project_dir / f"{run_name}_{run.id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Exporting run: {run.name} ({run.id}) state={run.state}")

    config = {k: v for k, v in run.config.items() if not str(k).startswith("_")}
    summary = dict(run.summary)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    try:
        hist = run.history(samples=100000)
        hist.to_csv(run_dir / "history.csv", index=False)
    except Exception as e:
        print(f"    history export failed: {repr(e)}")

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "url": run.url,
        **{
            f"summary/{k}": v
            for k, v in summary.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        },
        **{
            f"config/{k}": v
            for k, v in config.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        },
    }


def export_project(api: wandb.Api, project_name: str):
    print(f"\n=== Export project: {ENTITY}/{project_name} ===")

    project_dir = OUT_DIR / safe_name(project_name)
    project_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    try:
        runs = api.runs(f"{ENTITY}/{project_name}")
        for run in runs:
            rows.append(export_run(run, project_dir))
    except Exception as e:
        print(f"  failed to export project {project_name}: {repr(e)}")
        return

    if rows:
        index_path = project_dir / "_runs_index.csv"
        pd.DataFrame(rows).to_csv(index_path, index=False)
        print(f"  saved index: {index_path}")
    else:
        print("  no runs found")


def main():
    if not ENTITY:
        raise SystemExit("Set WANDB_ENTITY before exporting W&B runs.")

    api = wandb.Api()

    print("=== Current W&B viewer ===")
    print(api.viewer)

    print(f"\n=== Listing projects under entity: {ENTITY} ===")
    projects = list(api.projects(entity=ENTITY))

    project_names = [p.name for p in projects]
    for name in project_names:
        print(" -", name)

    matched = [
        name for name in project_names
        if "cs336" in name.lower()
    ]

    print("\n=== Matched CS336 projects ===")
    for name in matched:
        print(" -", name)

    if not matched:
        print("\nNo CS336 projects found under entity:", ENTITY)
        return

    for project_name in matched:
        export_project(api, project_name)

    print("\nDone. Exported files are under:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
