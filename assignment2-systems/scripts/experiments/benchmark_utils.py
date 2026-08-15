from __future__ import annotations

import json
import re
import subprocess
import timeit
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_root() -> Path:
    return project_root() / "experiments" / "results"


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def compact_range(values: list[int]) -> str:
    if not values:
        return "none"
    values = sorted(values)
    if len(values) == 1:
        return str(values[0])
    return f"{values[0]}-{values[-1]}"


def git_metadata() -> dict[str, Any]:
    root = project_root()
    metadata: dict[str, Any] = {
        "git_short_commit": None,
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return metadata

    metadata.update(
        {
            "git_short_commit": commit[:8],
        }
    )
    return metadata


def build_run_id(parts: list[str], git_short_commit: str | None = None, timestamp: str | None = None) -> str:
    semantic = slugify("_".join(part for part in parts if part))
    suffixes = []
    if git_short_commit:
        suffixes.append(git_short_commit)
    suffixes.append(timestamp or datetime.now().strftime("%Y%m%d-%H%M%S"))
    return slugify("_".join([semantic, *suffixes]))


def create_run_dir(experiment_name: str, run_id: str, results_root: Path | None = None) -> Path:
    root = results_root or default_results_root()
    base = root / slugify(experiment_name)
    candidate = base / slugify(run_id)
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    for attempt in range(2, 100):
        numbered = base / f"{slugify(run_id)}_r{attempt:02d}"
        if not numbered.exists():
            numbered.mkdir(parents=True, exist_ok=False)
            return numbered
    raise RuntimeError(f"Unable to allocate run directory under {base}")


def bytes_to_gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**3)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda_state(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sync_if_cuda(device)


def peak_memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return bytes_to_gib(torch.cuda.max_memory_allocated(device))


def time_ms(fn: Callable[[], None], device: torch.device, warmup_steps: int, timing_steps: int) -> tuple[float, float]:
    for _ in range(warmup_steps):
        fn()
        sync_if_cuda(device)

    measurements = []
    for _ in range(timing_steps):
        sync_if_cuda(device)
        start = timeit.default_timer()
        fn()
        sync_if_cuda(device)
        measurements.append(1000 * (timeit.default_timer() - start))

    if len(measurements) == 1:
        return measurements[0], 0.0
    return mean(measurements), stdev(measurements)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_gib": bytes_to_gib(props.total_memory),
                "cuda_runtime": torch.version.cuda,
            }
        )
    metadata.update(git_metadata())
    return metadata


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    public = dict(config)
    if "output_dir" in public:
        public["output_dir"] = _public_path(public["output_dir"])
    return public


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def _public_path(value: Any) -> str:
    path = Path(value)
    root = project_root()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return path.name


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def write_markdown_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No results.\n", encoding="utf-8")
        return

    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_report(
    path: Path,
    title: str,
    run_id: str,
    config: dict[str, Any],
    environment: dict[str, Any],
    outputs: list[str],
    notes: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Run",
        "",
        f"- Run ID: `{run_id}`",
        f"- Git commit: `{environment.get('git_short_commit') or 'unknown'}`",
        "",
        "## Configuration",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in sorted(config.items()):
        lines.append(f"| `{key}` | `{_format_cell(to_jsonable(value))}` |")

    lines.extend(
        [
            "",
            "## Environment",
            "",
            "| Key | Value |",
            "| --- | --- |",
        ]
    )
    for key, value in sorted(environment.items()):
        lines.append(f"| `{key}` | `{_format_cell(to_jsonable(value))}` |")

    lines.extend(["", "## Outputs", ""])
    for output in outputs:
        lines.append(f"- `{output}`")

    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
