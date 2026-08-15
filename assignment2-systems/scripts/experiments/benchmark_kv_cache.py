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
from cs336_basics.model import BasicsTransformerLM  # noqa: E402
from cs336_systems.kv_cache.cache import CachedTransformerLM, generate_greedy_no_cache  # noqa: E402


@dataclass
class KVCacheBenchmarkResult:
    implementation: str
    batch_size: int
    prompt_len: int
    max_new_tokens: int
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    dtype: str
    mean_ms: float | None
    std_ms: float | None
    mean_ms_per_new_token: float | None
    peak_gib: float | None
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark no-cache vs hand-written KV-cache generation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--max-new-tokens", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=default_results_root())
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-label", type=str, default=None)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> BasicsTransformerLM:
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10_000.0,
    )
    model.eval()
    model.to(device=device, dtype=dtype)
    return model


def make_prompt(args: argparse.Namespace, prompt_len: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, args.vocab_size, (args.batch_size, prompt_len), device=device)


def benchmark_one(
    implementation: str,
    model: BasicsTransformerLM,
    prompt: torch.Tensor,
    args: argparse.Namespace,
    dtype_name: str,
    device: torch.device,
    max_new_tokens: int,
) -> KVCacheBenchmarkResult:
    try:
        cached_model = CachedTransformerLM(model)

        @torch.no_grad()
        def step() -> None:
            if implementation == "no_cache":
                out = generate_greedy_no_cache(model, prompt, max_new_tokens=max_new_tokens)
            elif implementation == "kv_cache":
                out = cached_model.generate_greedy(prompt, max_new_tokens=max_new_tokens)
            else:
                raise ValueError(f"Unknown implementation: {implementation}")
            del out

        clear_cuda_state(device)
        mean_ms, std_ms = time_ms(step, device, args.warmup_steps, args.timing_steps)
        peak = peak_memory_gib(device)
        return KVCacheBenchmarkResult(
            implementation=implementation,
            batch_size=args.batch_size,
            prompt_len=prompt.shape[-1],
            max_new_tokens=max_new_tokens,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dtype=dtype_name,
            mean_ms=mean_ms,
            std_ms=std_ms,
            mean_ms_per_new_token=mean_ms / max_new_tokens,
            peak_gib=peak,
            status="ok",
            error="",
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _error_result(implementation, prompt, args, dtype_name, max_new_tokens, str(exc).splitlines()[0], status="oom")
    except RuntimeError as exc:
        return _error_result(implementation, prompt, args, dtype_name, max_new_tokens, str(exc).splitlines()[0], status="error")
    finally:
        clear_cuda_state(device)


def _error_result(
    implementation: str,
    prompt: torch.Tensor,
    args: argparse.Namespace,
    dtype_name: str,
    max_new_tokens: int,
    error: str,
    status: str,
) -> KVCacheBenchmarkResult:
    return KVCacheBenchmarkResult(
        implementation=implementation,
        batch_size=args.batch_size,
        prompt_len=prompt.shape[-1],
        max_new_tokens=max_new_tokens,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dtype=dtype_name,
        mean_ms=None,
        std_ms=None,
        mean_ms_per_new_token=None,
        peak_gib=None,
        status=status,
        error=error,
    )


def write_csv(results: list[KVCacheBenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(KVCacheBenchmarkResult.__dataclass_fields__))
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
            "kv-cache",
            args.run_label or "",
            f"b{args.batch_size}",
            f"prompt{compact_range(args.prompt_lengths)}",
            f"new{compact_range(args.max_new_tokens)}",
            f"d{args.d_model}",
            f"l{args.num_layers}",
            args.dtype,
            device.type,
        ],
        git_short_commit=environment.get("git_short_commit"),
    )
    output_dir = create_run_dir("kv_cache", args.run_id or generated_run_id, args.output_dir)
    config = public_config(vars(args) | {"device": str(device), "run_id": output_dir.name})
    write_json({"config": config, "environment": environment}, output_dir / "metadata.json")
    write_run_report(
        output_dir / "run.md",
        title="KV Cache Benchmark Run",
        run_id=output_dir.name,
        config=config,
        environment=environment,
        outputs=["metadata.json", "run.md", "results.csv", "results.json", "results.md"],
        notes=[
            "no_cache recomputes the full prefix for every generated token.",
            "kv_cache uses the hand-written cs336_systems.kv_cache wrapper.",
        ],
    )

    model = make_model(args, dtype, device)
    results: list[KVCacheBenchmarkResult] = []
    for prompt_len in args.prompt_lengths:
        if prompt_len + max(args.max_new_tokens) > args.context_length:
            print(f"[kv-cache] skip prompt={prompt_len}: prompt + max_new_tokens exceeds context_length")
            continue
        prompt = make_prompt(args, prompt_len, device)
        for max_new_tokens in args.max_new_tokens:
            for implementation in ("no_cache", "kv_cache"):
                print(f"[kv-cache] impl={implementation} prompt={prompt_len} new={max_new_tokens}")
                result = benchmark_one(implementation, model, prompt, args, args.dtype, device, max_new_tokens)
                results.append(result)
                print(
                    f"  {result.status}: "
                    f"total={result.mean_ms} ms "
                    f"per_token={result.mean_ms_per_new_token} ms "
                    f"peak={result.peak_gib} GiB"
                )

    rows = [asdict(result) for result in results]
    write_csv(results, output_dir / "results.csv")
    write_json(rows, output_dir / "results.json")
    write_markdown_table(rows, output_dir / "results.md")
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
