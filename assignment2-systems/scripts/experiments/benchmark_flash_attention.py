from __future__ import annotations

import argparse
import csv
import sys
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
from cs336_systems.flash_attention.flash_att_pytorch import flash_attention_pytorch  # noqa: E402
from cs336_systems.flash_attention.flash_att_triton import flash_attention_triton  # noqa: E402


IMPLEMENTATIONS = ("torch", "torch_compile", "flash_pytorch", "flash_triton")


@dataclass
class AttentionBenchmarkResult:
    implementation: str
    batch_size: int
    seq_len: int
    d_model: int
    dtype: str
    causal: bool
    forward_mean_ms: float | None
    forward_std_ms: float | None
    backward_mean_ms: float | None
    backward_std_ms: float | None
    forward_peak_gib: float | None
    backward_peak_gib: float | None
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations used in assignment2.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--d-models", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--implementations", nargs="+", default=list(IMPLEMENTATIONS), choices=IMPLEMENTATIONS)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timing-steps", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=default_results_root())
    parser.add_argument("--run-id", type=str, default=None, help="Optional semantic run id. A unique suffix is added only on collision.")
    parser.add_argument("--run-label", type=str, default=None, help="Optional human label to include in the generated run id.")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    grad_out = torch.randn_like(q)
    return q, k, v, grad_out


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    return positions[:, None] >= positions[None, :]


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        mask = make_causal_mask(q.shape[-2], q.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def build_attention_fn(name: str):
    if name == "torch":
        return torch_attention
    if name == "torch_compile":
        return torch.compile(torch_attention)
    if name == "flash_pytorch":
        return flash_attention_pytorch.apply
    if name == "flash_triton":
        return flash_attention_triton.apply
    raise ValueError(f"Unknown implementation: {name}")


def zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def benchmark_one(
    implementation: str,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    causal: bool,
    warmup_steps: int,
    timing_steps: int,
) -> AttentionBenchmarkResult:
    if implementation == "flash_triton" and device.type != "cuda":
        return _skipped_result(implementation, batch_size, seq_len, d_model, dtype_name, causal, "Triton requires CUDA.")

    attention_fn = build_attention_fn(implementation)
    try:
        q, k, v, grad_out = make_inputs(batch_size, seq_len, d_model, dtype, device)

        def forward_step(q_: torch.Tensor = q, k_: torch.Tensor = k, v_: torch.Tensor = v) -> None:
            out = attention_fn(q_, k_, v_, causal)
            del out

        clear_cuda_state(device)
        forward_mean, forward_std = time_ms(forward_step, device, warmup_steps, timing_steps)
        forward_peak = peak_memory_gib(device)

        def backward_step(
            q_: torch.Tensor = q,
            k_: torch.Tensor = k,
            v_: torch.Tensor = v,
            grad_out_: torch.Tensor = grad_out,
        ) -> None:
            zero_grads(q_, k_, v_)
            out = attention_fn(q_, k_, v_, causal)
            out.backward(grad_out_)
            del out

        clear_cuda_state(device)
        backward_mean, backward_std = time_ms(backward_step, device, warmup_steps, timing_steps)
        backward_peak = peak_memory_gib(device)

        return AttentionBenchmarkResult(
            implementation=implementation,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            dtype=dtype_name,
            causal=causal,
            forward_mean_ms=forward_mean,
            forward_std_ms=forward_std,
            backward_mean_ms=backward_mean,
            backward_std_ms=backward_std,
            forward_peak_gib=forward_peak,
            backward_peak_gib=backward_peak,
            status="ok",
            error="",
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _skipped_result(implementation, batch_size, seq_len, d_model, dtype_name, causal, str(exc).splitlines()[0], status="oom")
    except RuntimeError as exc:
        return _skipped_result(implementation, batch_size, seq_len, d_model, dtype_name, causal, str(exc).splitlines()[0], status="error")
    finally:
        clear_cuda_state(device)


def _skipped_result(
    implementation: str,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype_name: str,
    causal: bool,
    error: str,
    status: str = "skipped",
) -> AttentionBenchmarkResult:
    return AttentionBenchmarkResult(
        implementation=implementation,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        dtype=dtype_name,
        causal=causal,
        forward_mean_ms=None,
        forward_std_ms=None,
        backward_mean_ms=None,
        backward_std_ms=None,
        forward_peak_gib=None,
        backward_peak_gib=None,
        status=status,
        error=error,
    )


def write_csv(results: list[AttentionBenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(AttentionBenchmarkResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    environment = environment_metadata(device)
    generated_run_id = build_run_id(
        [
            "flash-attention",
            args.run_label or "",
            f"impl-{'+'.join(args.implementations)}",
            f"b{args.batch_size}",
            f"seq{compact_range(args.seq_lengths)}",
            f"d{compact_range(args.d_models)}",
            args.dtype,
            "causal" if args.causal else "noncausal",
            device.type,
        ],
        git_short_commit=environment.get("git_short_commit"),
    )
    run_id = args.run_id or generated_run_id
    output_dir = create_run_dir("attention", run_id, args.output_dir)

    config = public_config(vars(args) | {"device": str(device), "run_id": output_dir.name})
    write_json({"config": config, "environment": environment}, output_dir / "metadata.json")
    write_run_report(
        output_dir / "run.md",
        title="Flash Attention Benchmark Run",
        run_id=output_dir.name,
        config=config,
        environment=environment,
        outputs=["metadata.json", "run.md", "results.csv", "results.json", "results.md"],
        notes=[
            "CSV and JSON are canonical machine-readable records.",
            "Markdown results are a quick preview and should not be used as the source for plots.",
        ],
    )

    results: list[AttentionBenchmarkResult] = []
    for seq_len in args.seq_lengths:
        for d_model in args.d_models:
            for implementation in args.implementations:
                print(f"[attention] impl={implementation} batch={args.batch_size} seq={seq_len} d={d_model}")
                result = benchmark_one(
                    implementation=implementation,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    d_model=d_model,
                    dtype=dtype,
                    dtype_name=args.dtype,
                    device=device,
                    causal=args.causal,
                    warmup_steps=args.warmup_steps,
                    timing_steps=args.timing_steps,
                )
                results.append(result)
                print(f"  {result.status}: fwd={result.forward_mean_ms} ms bwd={result.backward_mean_ms} ms")

    rows = [asdict(result) for result in results]
    write_csv(results, output_dir / "results.csv")
    write_json(rows, output_dir / "results.json")
    write_markdown_table(rows, output_dir / "results.md")
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
