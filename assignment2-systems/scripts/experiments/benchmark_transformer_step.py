from __future__ import annotations

import argparse
import csv
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from benchmark_utils import (  # noqa: E402
    build_run_id,
    clear_cuda_state,
    compact_range,
    create_run_dir,
    default_results_root,
    environment_metadata,
    peak_memory_gib,
    public_config,
    time_ms,
    write_json,
    write_markdown_table,
    write_run_report,
)
from cs336_basics.model import BasicsTransformerLM  # noqa: E402


MODEL_CONFIGS = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
}
PRECISIONS = ("fp32", "bf16", "fp16")
MODES = ("forward", "forward_backward", "train_step")


@dataclass
class TransformerStepResult:
    model_size: str
    mode: str
    precision: str
    batch_size: int
    context_length: int
    vocab_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    num_parameters_m: float
    mean_ms: float | None
    std_ms: float | None
    peak_gib: float | None
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Transformer LM training-step phases.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-sizes", nargs="+", choices=list(MODEL_CONFIGS), default=["tiny"])
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=["fp32"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timing-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--output-dir", type=Path, default=default_results_root())
    parser.add_argument("--run-id", type=str, default=None, help="Optional semantic run id. A unique suffix is added only on collision.")
    parser.add_argument("--run-label", type=str, default=None, help="Optional human label to include in the generated run id.")
    return parser.parse_args()


def precision_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"Unknown precision: {precision}")


def autocast_context(device: torch.device, precision: str):
    dtype = precision_dtype(precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_model(model_size: str, vocab_size: int, context_length: int, rope_theta: float, device: torch.device) -> BasicsTransformerLM:
    config = MODEL_CONFIGS[model_size]
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=rope_theta,
    ).to(device)


def make_batch(batch_size: int, context_length: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return input_ids, labels


def compute_loss(model: BasicsTransformerLM, input_ids: torch.Tensor, labels: torch.Tensor, vocab_size: int, device: torch.device, precision: str):
    with autocast_context(device, precision):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size).float(), labels.reshape(-1))
    return loss


def benchmark_one(
    model_size: str,
    mode: str,
    precision: str,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    warmup_steps: int,
    timing_steps: int,
    learning_rate: float,
    rope_theta: float,
    device: torch.device,
) -> TransformerStepResult:
    model_config = MODEL_CONFIGS[model_size]
    model = optimizer = input_ids = labels = None
    num_parameters_m = 0.0

    try:
        model = make_model(model_size, vocab_size, context_length, rope_theta, device)
        model.train()
        num_parameters_m = sum(parameter.numel() for parameter in model.parameters()) / 1_000_000
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        input_ids, labels = make_batch(batch_size, context_length, vocab_size, device)

        def forward_step(
            model_: BasicsTransformerLM = model,
            optimizer_: torch.optim.Optimizer = optimizer,
            input_ids_: torch.Tensor = input_ids,
            labels_: torch.Tensor = labels,
        ) -> None:
            optimizer_.zero_grad(set_to_none=True)
            loss = compute_loss(model_, input_ids_, labels_, vocab_size, device, precision)
            del loss

        def forward_backward_step(
            model_: BasicsTransformerLM = model,
            optimizer_: torch.optim.Optimizer = optimizer,
            input_ids_: torch.Tensor = input_ids,
            labels_: torch.Tensor = labels,
        ) -> None:
            optimizer_.zero_grad(set_to_none=True)
            loss = compute_loss(model_, input_ids_, labels_, vocab_size, device, precision)
            loss.backward()
            del loss

        def train_step(
            model_: BasicsTransformerLM = model,
            optimizer_: torch.optim.Optimizer = optimizer,
            input_ids_: torch.Tensor = input_ids,
            labels_: torch.Tensor = labels,
        ) -> None:
            optimizer_.zero_grad(set_to_none=True)
            loss = compute_loss(model_, input_ids_, labels_, vocab_size, device, precision)
            loss.backward()
            optimizer_.step()
            del loss

        step_fn = {
            "forward": forward_step,
            "forward_backward": forward_backward_step,
            "train_step": train_step,
        }[mode]

        clear_cuda_state(device)
        mean_ms, std_ms = time_ms(step_fn, device, warmup_steps, timing_steps)
        peak_gib = peak_memory_gib(device)

        return TransformerStepResult(
            model_size=model_size,
            mode=mode,
            precision=precision,
            batch_size=batch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            d_model=model_config["d_model"],
            d_ff=model_config["d_ff"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            num_parameters_m=num_parameters_m,
            mean_ms=mean_ms,
            std_ms=std_ms,
            peak_gib=peak_gib,
            status="ok",
            error="",
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _failed_result(model_size, mode, precision, batch_size, context_length, vocab_size, num_parameters_m, str(exc).splitlines()[0], "oom")
    except RuntimeError as exc:
        return _failed_result(model_size, mode, precision, batch_size, context_length, vocab_size, num_parameters_m, str(exc).splitlines()[0], "error")
    finally:
        del model, optimizer, input_ids, labels
        clear_cuda_state(device)


def _failed_result(
    model_size: str,
    mode: str,
    precision: str,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    num_parameters_m: float,
    error: str,
    status: str,
) -> TransformerStepResult:
    model_config = MODEL_CONFIGS[model_size]
    return TransformerStepResult(
        model_size=model_size,
        mode=mode,
        precision=precision,
        batch_size=batch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        num_parameters_m=num_parameters_m,
        mean_ms=None,
        std_ms=None,
        peak_gib=None,
        status=status,
        error=error,
    )


def write_csv(results: list[TransformerStepResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TransformerStepResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    environment = environment_metadata(device)
    generated_run_id = build_run_id(
        [
            "transformer-step",
            args.run_label or "",
            f"models-{'+'.join(args.model_sizes)}",
            f"b{args.batch_size}",
            f"ctx{compact_range(args.context_lengths)}",
            f"prec-{'+'.join(args.precisions)}",
            f"modes-{'+'.join(args.modes)}",
            device.type,
        ],
        git_short_commit=environment.get("git_short_commit"),
    )
    output_dir = create_run_dir("transformer-step", args.run_id or generated_run_id, args.output_dir)
    config = public_config(vars(args) | {"device": str(device), "run_id": output_dir.name})
    write_json({"config": config, "environment": environment}, output_dir / "metadata.json")
    write_run_report(
        output_dir / "run.md",
        title="Transformer Step Benchmark Run",
        run_id=output_dir.name,
        config=config,
        environment=environment,
        outputs=["metadata.json", "run.md", "results.csv", "results.json", "results.md"],
        notes=[
            "Forward mode includes logits and cross-entropy loss construction.",
            "Forward-backward mode excludes optimizer.step().",
            "Train-step mode includes AdamW step; warmup allocates AdamW state before timed iterations.",
        ],
    )

    results: list[TransformerStepResult] = []
    for model_size in args.model_sizes:
        for context_length in args.context_lengths:
            for precision in args.precisions:
                for mode in args.modes:
                    print(f"[transformer] model={model_size} ctx={context_length} precision={precision} mode={mode}", flush=True)
                    result = benchmark_one(
                        model_size=model_size,
                        mode=mode,
                        precision=precision,
                        batch_size=args.batch_size,
                        context_length=context_length,
                        vocab_size=args.vocab_size,
                        warmup_steps=args.warmup_steps,
                        timing_steps=args.timing_steps,
                        learning_rate=args.learning_rate,
                        rope_theta=args.rope_theta,
                        device=device,
                    )
                    results.append(result)
                    rows = [asdict(item) for item in results]
                    write_csv(results, output_dir / "results.csv")
                    write_json(rows, output_dir / "results.json")
                    write_markdown_table(rows, output_dir / "results.md")
                    print(f"  {result.status}: mean={result.mean_ms} ms peak={result.peak_gib} GiB", flush=True)

    rows = [asdict(result) for result in results]
    write_csv(results, output_dir / "results.csv")
    write_json(rows, output_dir / "results.json")
    write_markdown_table(rows, output_dir / "results.md")
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
