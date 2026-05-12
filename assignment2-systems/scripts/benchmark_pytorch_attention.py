from __future__ import annotations

import argparse
import csv
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CS336_BASICS = REPO_ROOT / "assignment2-systems" / "cs336-basics"
sys.path.append(str(CS336_BASICS))

from cs336_basics.model import scaled_dot_product_attention  # noqa: E402


BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENGTHS = [256, 1024, 4096, 8192, 16384]


@dataclass
class AttentionResult:
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
    parser = argparse.ArgumentParser(description="Benchmark vanilla PyTorch scaled dot-product attention.")
    parser.add_argument("--device", default="cuda", help="Device to run on. This benchmark is intended for CUDA.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--d-models", type=int, nargs="+", default=D_MODELS)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=SEQ_LENGTHS)
    parser.add_argument("--causal", action="store_true", help="Use a lower-triangular causal attention mask.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/pytorch_attention"))
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


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)


def zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def time_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    device: torch.device,
    warmup_steps: int,
    timing_steps: int,
) -> float:
    for _ in range(warmup_steps):
        out = attention(q, k, v, mask)
        sync_if_cuda(device)
        del out

    times = []
    for _ in range(timing_steps):
        sync_if_cuda(device)
        start = timeit.default_timer()
        out = attention(q, k, v, mask)
        sync_if_cuda(device)
        times.append(timeit.default_timer() - start)
        del out
    return 1000 * sum(times) / len(times)


def time_backward(
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
        out = attention(q, k, v, mask)
        grad_out = torch.randn_like(out)
        sync_if_cuda(device)
        out.backward(grad_out)
        sync_if_cuda(device)
        del out, grad_out

    memory_before_backward = 0
    times = []
    for step in range(timing_steps):
        zero_grads(q, k, v)
        out = attention(q, k, v, mask)
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
    """Estimate autograd-saved attention intermediates for vanilla attention.

    Vanilla attention materializes scores S and probabilities P with shape
    (batch, seq, seq). PyTorch autograd must also keep enough tensor values to
    compute gradients through the two matmuls and softmax. A useful lower-bound
    estimate for the large term is S + P plus Q/K/V and output activations.
    """

    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    qkv_and_output = 4 * batch_size * seq_len * d_model
    scores_and_probs = 2 * batch_size * seq_len * seq_len
    return bytes_to_gib((qkv_and_output + scores_and_probs) * bytes_per_element)


def benchmark_config(
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    causal: bool,
    warmup_steps: int,
    timing_steps: int,
) -> AttentionResult:
    estimated_saved_gib = estimate_saved_for_backward_gib(BATCH_SIZE, seq_len, d_model, dtype)
    q = k = v = mask = None

    try:
        q, k, v = make_inputs(BATCH_SIZE, seq_len, d_model, dtype, device)
        mask = make_mask(seq_len, device, causal)

        clear_cuda_state(device)
        forward_ms = time_forward(q, k, v, mask, device, warmup_steps, timing_steps)

        clear_cuda_state(device)
        backward_ms, memory_before_backward = time_backward(q, k, v, mask, device, warmup_steps, timing_steps)

        return AttentionResult(
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
        return AttentionResult(
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
    finally:
        del q, k, v, mask
        clear_cuda_state(device)


def write_csv(results: list[AttentionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(AttentionResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_markdown(results: list[AttentionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
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


def print_result(result: AttentionResult) -> None:
    if result.status == "ok":
        print(
            f"batch={result.batch_size} seq={result.seq_len:5d} d={result.d_model:3d} "
            f"fwd={result.forward_mean_ms:.3f} ms bwd={result.backward_mean_ms:.3f} ms "
            f"mem_before_bwd={result.memory_before_backward_gib:.3f} GiB"
        )
    else:
        print(
            f"batch={result.batch_size} seq={result.seq_len:5d} d={result.d_model:3d} "
            f"status=OOM est_saved={result.estimated_saved_for_backward_gib:.3f} GiB"
        )


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

    results: list[AttentionResult] = []
    for d_model, seq_len in ((d, n) for d in args.d_models for n in args.seq_lengths):
        print(f"\n[benchmark] batch={BATCH_SIZE}, seq_len={seq_len}, d_model={d_model}, dtype={args.dtype}")
        result = benchmark_config(
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
    csv_path = args.output_dir / f"pytorch_attention_{args.dtype}_{suffix}.csv"
    md_path = args.output_dir / f"pytorch_attention_{args.dtype}_{suffix}.md"
    write_csv(results, csv_path)
    write_markdown(results, md_path)
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved Markdown table to {md_path}")


if __name__ == "__main__":
    main()
