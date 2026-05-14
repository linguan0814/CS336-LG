from __future__ import annotations

import json
import platform
import timeit
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_root() -> Path:
    return project_root() / "experiments" / "results"


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
        "python": platform.python_version(),
        "platform": platform.platform(),
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
    return metadata


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


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
