from __future__ import annotations

import argparse
import csv
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CS336_BASICS = REPO_ROOT / "assignment2-systems" / "cs336-basics"
sys.path.append(str(CS336_BASICS))

from cs336_basics.model import scaled_dot_product_attention  # noqa: E402


BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]


AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


@dataclass
class JITAttentionResult:
    implementation: str
    batch_size: int
    seq_len: int
    d_model: int
    dtype: str
    forward_mean_ms: float | None
    backward_mean_ms: float | None
    memory_before_backward_gib: float | None
    estimated_saved_for_backward_gib: float
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eager vs torch.compile PyTorch attention.")
    parser.add_argument("--device", default="cuda", help="Device to run on. This benchmark is intended for CUDA.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--d-models", type=int, nargs="+", default=D_MODELS)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=SEQ_LENGTHS)
    parser.add_argument("--causal", action="store_true", help="Use a lower-triangular causal attention mask.")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["eager", "compiled"],
        choices=["eager", "compiled"],
        help="Which implementations to benchmark.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/jit_attention"))
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


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


def make_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    return q, k, v


def make_mask(seq_len: int, device: torch.device, causal: bool) -> torch.Tensor | None:
    if not causal:
        return None
    positions = torch.arange(seq_len, device=device)
    return positions[:, None] >= positions[None, :]


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)


def build_attention_fn(name: str) -> AttentionFn:
    if name == "eager":
        return eager_attention
    if name == "compiled":
        return torch.compile(eager_attention)
    raise ValueError(f"Unknown implementation: {name}")


def zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def time_forward(
    attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    device: torch.device,
    warmup_steps: int,
    timing_steps: int,
) -> float:
    for _ in range(warmup_steps):
        out = attention_fn(q, k, v, mask)
        sync_if_cuda(device)
        del out

    times = []
    for _ in range(timing_steps):
        sync_if_cuda(device)
        start = timeit.default_timer()
        out = attention_fn(q, k, v, mask)
        sync_if_cuda(device)
        times.append(timeit.default_timer() - start)
        del out
    return 1000 * sum(times) / len(times)


def time_backward(
    attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    device: torch.device,
    warmup_steps: int,
    timing_steps: int,
) -> tuple[float, int]:
    for _ in range(warmup_steps):
        zero_grads(q, k, v)
        out = attention_fn(q, k, v, mask)
        grad_out = torch.randn_like(out)
        sync_if_cuda(device)
        out.backward(grad_out)
        sync_if_cuda(device)
        del out, grad_out

    memory_before_backward = 0
    times = []
    for step in range(timing_steps):
        zero_grads(q, k, v)
        out = attention_fn(q, k, v, mask)
        grad_out = torch.randn_like(out)
        sync_if_cuda(device)

        if step == 0 and device.type == "cuda":
            memory_before_backward = torch.cuda.memory_allocated(device)

        start = timeit.default_timer()
        out.backward(grad_out)
        sync_if_cuda(device)
        times.append(timeit.default_timer() - start)
        del out, grad_out

    return 1000 * sum(times) / len(times), memory_before_backward


def estimate_saved_for_backward_gib(batch_size: int, seq_len: int, d_model: int, dtype: torch.dtype) -> float:
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    qkv_and_output = 4 * batch_size * seq_len * d_model
    scores_and_probs = 2 * batch_size * seq_len * seq_len
    return bytes_to_gib((qkv_and_output + scores_and_probs) * bytes_per_element)


def benchmark_config(
    implementation: str,
    attention_fn: AttentionFn,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    causal: bool,
    warmup_steps: int,
    timing_steps: int,
) -> JITAttentionResult:
    estimated_saved_gib = estimate_saved_for_backward_gib(BATCH_SIZE, seq_len, d_model, dtype)
    q = k = v = mask = None

    try:
        q, k, v = make_inputs(BATCH_SIZE, seq_len, d_model, dtype, device)
        mask = make_mask(seq_len, device, causal)

        clear_cuda_state(device)
        forward_ms = time_forward(attention_fn, q, k, v, mask, device, warmup_steps, timing_steps)

        clear_cuda_state(device)
        backward_ms, memory_before_backward = time_backward(
            attention_fn, q, k, v, mask, device, warmup_steps, timing_steps
        )

        return JITAttentionResult(
            implementation=implementation,
            batch_size=BATCH_SIZE,
            seq_len=seq_len,
            d_model=d_model,
            dtype=dtype_name,
            forward_mean_ms=forward_ms,
            backward_mean_ms=backward_ms,
            memory_before_backward_gib=bytes_to_gib(memory_before_backward),
            estimated_saved_for_backward_gib=estimated_saved_gib,
            status="ok",
            error="",
        )
    except torch.cuda.OutOfMemoryError as exc:
        return JITAttentionResult(
            implementation=implementation,
            batch_size=BATCH_SIZE,
            seq_len=seq_len,
            d_model=d_model,
            dtype=dtype_name,
            forward_mean_ms=None,
            backward_mean_ms=None,
            memory_before_backward_gib=None,
            estimated_saved_for_backward_gib=estimated_saved_gib,
            status="oom",
            error=str(exc).splitlines()[0],
        )
    except RuntimeError as exc:
        return JITAttentionResult(
            implementation=implementation,
            batch_size=BATCH_SIZE,
            seq_len=seq_len,
            d_model=d_model,
            dtype=dtype_name,
            forward_mean_ms=None,
            backward_mean_ms=None,
            memory_before_backward_gib=None,
            estimated_saved_for_backward_gib=estimated_saved_gib,
            status="error",
            error=str(exc).splitlines()[0],
        )
    finally:
        del q, k, v, mask
        clear_cuda_state(device)


def write_csv(results: list[JITAttentionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(JITAttentionResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_markdown(results: list[JITAttentionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "impl",
        "batch",
        "seq_len",
        "d_model",
        "dtype",
        "fwd ms",
        "bwd ms",
        "mem before bwd GiB",
        "est. saved GiB",
        "status",
    ]

    def fmt(value: float | None) -> str:
        return "OOM" if value is None else f"{value:.3f}"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.implementation,
                    str(r.batch_size),
                    str(r.seq_len),
                    str(r.d_model),
                    r.dtype,
                    fmt(r.forward_mean_ms),
                    fmt(r.backward_mean_ms),
                    fmt(r.memory_before_backward_gib),
                    f"{r.estimated_saved_for_backward_gib:.3f}",
                    r.status,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_result(result: JITAttentionResult) -> None:
    if result.status == "ok":
        print(
            f"{result.implementation:8s} batch={result.batch_size} seq={result.seq_len:5d} "
            f"d={result.d_model:3d} fwd={result.forward_mean_ms:.3f} ms "
            f"bwd={result.backward_mean_ms:.3f} ms "
            f"mem_before_bwd={result.memory_before_backward_gib:.3f} GiB"
        )
    else:
        print(
            f"{result.implementation:8s} batch={result.batch_size} seq={result.seq_len:5d} "
            f"d={result.d_model:3d} status={result.status.upper()} "
            f"est_saved={result.estimated_saved_for_backward_gib:.3f} GiB"
        )


def write_summary(results: list[JITAttentionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[int, int, str], dict[str, JITAttentionResult]] = {}
    for result in results:
        by_key.setdefault((result.seq_len, result.d_model, result.dtype), {})[result.implementation] = result

    headers = [
        "seq_len",
        "d_model",
        "dtype",
        "eager fwd ms",
        "compiled fwd ms",
        "fwd speedup",
        "eager bwd ms",
        "compiled bwd ms",
        "bwd speedup",
        "status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    def fmt(value: float | None) -> str:
        return "OOM" if value is None else f"{value:.3f}"

    for (seq_len, d_model, dtype), impls in sorted(by_key.items()):
        eager = impls.get("eager")
        compiled = impls.get("compiled")
        eager_fwd = eager.forward_mean_ms if eager else None
        compiled_fwd = compiled.forward_mean_ms if compiled else None
        eager_bwd = eager.backward_mean_ms if eager else None
        compiled_bwd = compiled.backward_mean_ms if compiled else None
        fwd_speedup = eager_fwd / compiled_fwd if eager_fwd and compiled_fwd else None
        bwd_speedup = eager_bwd / compiled_bwd if eager_bwd and compiled_bwd else None
        status = ",".join(f"{name}:{result.status}" for name, result in sorted(impls.items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seq_len),
                    str(d_model),
                    dtype,
                    fmt(eager_fwd),
                    fmt(compiled_fwd),
                    fmt(fwd_speedup),
                    fmt(eager_bwd),
                    fmt(compiled_bwd),
                    fmt(bwd_speedup),
                    status,
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        total_memory_gib = bytes_to_gib(torch.cuda.get_device_properties(device).total_memory)
        print(f"Running on {gpu_name} ({total_memory_gib:.2f} GiB)")

    attention_fns = {name: build_attention_fn(name) for name in args.implementations}
    results: list[JITAttentionResult] = []

    for d_model, seq_len in ((d, n) for d in args.d_models for n in args.seq_lengths):
        for implementation, attention_fn in attention_fns.items():
            print(
                f"\n[benchmark] impl={implementation}, batch={BATCH_SIZE}, "
                f"seq_len={seq_len}, d_model={d_model}, dtype={args.dtype}"
            )
            result = benchmark_config(
                implementation=implementation,
                attention_fn=attention_fn,
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
            print_result(result)

    suffix = "causal" if args.causal else "noncausal"
    impl_suffix = "_".join(args.implementations)
    csv_path = args.output_dir / f"jit_attention_{impl_suffix}_{args.dtype}_{suffix}.csv"
    md_path = args.output_dir / f"jit_attention_{impl_suffix}_{args.dtype}_{suffix}.md"
    summary_path = args.output_dir / f"jit_attention_summary_{impl_suffix}_{args.dtype}_{suffix}.md"
    write_csv(results, csv_path)
    write_markdown(results, md_path)
    write_summary(results, summary_path)
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved Markdown table to {md_path}")
    print(f"Saved speedup summary to {summary_path}")


if __name__ == "__main__":
    main()
