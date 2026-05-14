from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_utils import default_results_root, write_json, write_markdown_table


BYTES_FP32 = 4


@dataclass
class MemoryAccountingRow:
    strategy: str
    world_size: int
    parameters_m: float
    param_gib_per_rank: float
    grad_gib_per_rank: float
    optimizer_state_gib_per_rank: float
    total_gib_per_rank: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Theoretical per-rank memory accounting for DDP/ZeRO/FSDP.")
    parser.add_argument("--parameters-m", type=float, default=125.0, help="Number of model parameters in millions.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output-dir", type=Path, default=default_results_root() / "distributed_memory")
    return parser.parse_args()


def gib(num_parameters: float, bytes_per_parameter: int = BYTES_FP32) -> float:
    return num_parameters * bytes_per_parameter / (1024**3)


def accounting_rows(parameters_m: float, world_sizes: list[int]) -> list[MemoryAccountingRow]:
    num_parameters = parameters_m * 1_000_000
    rows: list[MemoryAccountingRow] = []
    for world_size in world_sizes:
        full_param = gib(num_parameters)
        full_grad = gib(num_parameters)
        adamw_state = 2 * gib(num_parameters)

        rows.append(
            MemoryAccountingRow(
                strategy="DDP",
                world_size=world_size,
                parameters_m=parameters_m,
                param_gib_per_rank=full_param,
                grad_gib_per_rank=full_grad,
                optimizer_state_gib_per_rank=adamw_state,
                total_gib_per_rank=full_param + full_grad + adamw_state,
                notes="Replicates parameters, gradients, and AdamW first/second moments.",
            )
        )
        rows.append(
            MemoryAccountingRow(
                strategy="ShardedOptimizer",
                world_size=world_size,
                parameters_m=parameters_m,
                param_gib_per_rank=full_param,
                grad_gib_per_rank=full_grad,
                optimizer_state_gib_per_rank=adamw_state / world_size,
                total_gib_per_rank=full_param + full_grad + adamw_state / world_size,
                notes="Replicates model and gradients; shards optimizer state.",
            )
        )
        rows.append(
            MemoryAccountingRow(
                strategy="FSDP",
                world_size=world_size,
                parameters_m=parameters_m,
                param_gib_per_rank=full_param / world_size,
                grad_gib_per_rank=full_grad / world_size,
                optimizer_state_gib_per_rank=adamw_state / world_size,
                total_gib_per_rank=(full_param + full_grad + adamw_state) / world_size,
                notes="Shards parameters, gradients, and optimizer state; ignores transient all-gather buffers.",
            )
        )
    return rows


def write_csv(rows: list[MemoryAccountingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MemoryAccountingRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    rows = accounting_rows(args.parameters_m, args.world_sizes)
    output_dir = args.output_dir / f"{args.parameters_m:g}M_params"
    row_dicts = [asdict(row) for row in rows]
    write_csv(rows, output_dir / "results.csv")
    write_json(row_dicts, output_dir / "results.json")
    write_markdown_table(row_dicts, output_dir / "results.md")
    write_json(vars(args), output_dir / "metadata.json")
    print(f"Saved memory accounting to {output_dir}")


if __name__ == "__main__":
    main()
