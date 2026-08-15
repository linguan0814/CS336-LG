from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_utils import default_results_root, write_markdown_table


ATTENTION_LABELS = {
    "torch": "PyTorch eager attention",
    "torch_compile": "torch.compile attention",
    "flash_pytorch": "FlashAttention, PyTorch tiled reference",
    "flash_triton": "FlashAttention, Triton forward + PyTorch backward",
}

MEMORY_LABELS = {
    "DDP": "DDP: replicated params/grads/optimizer",
    "ShardedOptimizer": "ZeRO-1 style: sharded optimizer state",
    "FSDP": "FSDP style: sharded params/grads/optimizer",
}

MODE_LABELS = {
    "forward": "Forward + loss",
    "forward_backward": "Forward + backward",
    "train_step": "Full train step (AdamW)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate assignment2 experiment CSVs and produce report figures.")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--output-dir", type=Path, default=default_results_root() / "analysis")
    return parser.parse_args()


def read_domain_results(results_root: Path, domain: str) -> pd.DataFrame:
    frames = []
    for csv_path in sorted((results_root / domain).glob("*/results.csv")):
        frame = pd.read_csv(csv_path)
        frame["run_id"] = csv_path.parent.name
        frame["domain"] = domain
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_attention(attention: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if attention.empty:
        return paths

    ok = attention[attention["status"] == "ok"].copy()
    core = ok[
        ok["run_id"].str.contains("primary-core|causal-core|primary-large|causal-large|long-context", regex=True)
        & ok["implementation"].isin(["torch", "torch_compile", "flash_triton"])
    ].copy()

    for causal_value, label in [(False, "noncausal"), (True, "causal")]:
        subset = core[(core["causal"] == causal_value) & (core["d_model"] == 64)]
        if subset.empty:
            continue

        plt.figure(figsize=(8, 5))
        for implementation, group in subset.groupby("implementation"):
            group = group.sort_values("seq_len")
            plt.plot(group["seq_len"], group["forward_mean_ms"], marker="o", label=ATTENTION_LABELS.get(implementation, implementation))
        plt.xscale("log", base=2)
        plt.xlabel("Sequence length")
        plt.ylabel("Forward mean latency (ms)")
        plt.title(f"Attention forward latency (batch=4, d_model=64, {label})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        path = output_dir / "figures" / f"attention_forward_d64_{label}.png"
        savefig(path)
        paths[f"attention_forward_d64_{label}"] = path

        plt.figure(figsize=(8, 5))
        for implementation, group in subset.groupby("implementation"):
            group = group.sort_values("seq_len")
            plt.plot(group["seq_len"], group["backward_mean_ms"], marker="o", label=ATTENTION_LABELS.get(implementation, implementation))
        plt.xscale("log", base=2)
        plt.xlabel("Sequence length")
        plt.ylabel("Backward mean latency (ms)")
        plt.title(f"Attention backward latency (batch=4, d_model=64, {label})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        path = output_dir / "figures" / f"attention_backward_d64_{label}.png"
        savefig(path)
        paths[f"attention_backward_d64_{label}"] = path

        speedup_rows = []
        for seq_len, group in subset.groupby("seq_len"):
            baseline = group[group["implementation"] == "torch"]["forward_mean_ms"]
            triton = group[group["implementation"] == "flash_triton"]["forward_mean_ms"]
            compiled = group[group["implementation"] == "torch_compile"]["forward_mean_ms"]
            if baseline.empty or triton.empty:
                continue
            speedup_rows.append(
                {
                    "seq_len": seq_len,
                    "flash_triton_forward_speedup_vs_torch": float(baseline.iloc[0] / triton.iloc[0]),
                    "torch_compile_forward_speedup_vs_torch": float(baseline.iloc[0] / compiled.iloc[0]) if not compiled.empty else None,
                    "causal": causal_value,
                }
            )
        if speedup_rows:
            speedup = pd.DataFrame(speedup_rows).sort_values("seq_len")
            speedup.to_csv(output_dir / f"attention_forward_speedup_d64_{label}.csv", index=False)

            plt.figure(figsize=(8, 5))
            plt.plot(speedup["seq_len"], speedup["flash_triton_forward_speedup_vs_torch"], marker="o", label="Triton FlashAttention forward vs PyTorch eager")
            if "torch_compile_forward_speedup_vs_torch" in speedup:
                plt.plot(speedup["seq_len"], speedup["torch_compile_forward_speedup_vs_torch"], marker="o", label="torch.compile vs PyTorch eager")
            plt.xscale("log", base=2)
            plt.xlabel("Sequence length")
            plt.ylabel("Forward speedup over PyTorch eager")
            plt.title(f"Attention forward speedup (batch=4, d_model=64, {label})")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            path = output_dir / "figures" / f"attention_forward_speedup_d64_{label}.png"
            savefig(path)
            paths[f"attention_forward_speedup_d64_{label}"] = path

    small = ok[ok["run_id"].str.contains("flash-pytorch-small|flash-pytorch-reference") & (ok["d_model"] == 64)]
    if not small.empty:
        plt.figure(figsize=(8, 5))
        for implementation, group in small.groupby("implementation"):
            group = group.sort_values("seq_len")
            plt.plot(group["seq_len"], group["forward_mean_ms"], marker="o", label=ATTENTION_LABELS.get(implementation, implementation))
        plt.yscale("log")
        plt.xscale("log", base=2)
        plt.xlabel("Sequence length")
        plt.ylabel("Forward mean latency (ms, log scale)")
        plt.title("Reference PyTorch tiled FlashAttention vs Triton kernel")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        path = output_dir / "figures" / "attention_flash_pytorch_small_forward_log.png"
        savefig(path)
        paths["attention_flash_pytorch_small_forward_log"] = path

    return paths


def plot_memory_accounting(memory: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if memory.empty:
        return paths

    for parameters_m, subset in memory.groupby("parameters_m"):
        plt.figure(figsize=(8, 5))
        for strategy, group in subset.groupby("strategy"):
            group = group.sort_values("world_size")
            plt.plot(group["world_size"], group["total_gib_per_rank"], marker="o", label=MEMORY_LABELS.get(strategy, strategy))
        plt.xlabel("World size")
        plt.ylabel("Persistent memory per rank (GiB)")
        plt.title(f"Distributed persistent memory, {parameters_m:g}M parameters")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        path = output_dir / "figures" / f"distributed_memory_{parameters_m:g}m.png"
        savefig(path)
        paths[f"distributed_memory_{parameters_m:g}m"] = path

    return paths


def plot_transformer_step(transformer: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if transformer.empty:
        return paths

    ok = transformer[transformer["status"] == "ok"].copy()
    for model_size, subset_model in ok.groupby("model_size"):
        for precision, subset in subset_model.groupby("precision"):
            plt.figure(figsize=(8, 5))
            for mode, group in subset.groupby("mode"):
                group = group.sort_values("context_length")
                plt.plot(group["context_length"], group["mean_ms"], marker="o", label=MODE_LABELS.get(mode, mode))
            batch_size = subset["batch_size"].iloc[0]
            parameters_m = subset["num_parameters_m"].max()
            plt.xscale("log", base=2)
            plt.xlabel("Context length")
            plt.ylabel("Mean latency (ms)")
            plt.title(f"Transformer step latency ({model_size}, {parameters_m:.1f}M params, {precision}, batch={batch_size})")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            path = output_dir / "figures" / f"transformer_step_{model_size}_{precision}.png"
            savefig(path)
            paths[f"transformer_step_{model_size}_{precision}"] = path

            if subset["peak_gib"].notna().any():
                plt.figure(figsize=(8, 5))
                for mode, group in subset.groupby("mode"):
                    group = group.sort_values("context_length")
                    plt.plot(group["context_length"], group["peak_gib"], marker="o", label=MODE_LABELS.get(mode, mode))
                plt.xscale("log", base=2)
                plt.xlabel("Context length")
                plt.ylabel("Peak allocated memory (GiB)")
                plt.title(f"Transformer step peak memory ({model_size}, {parameters_m:.1f}M params, {precision}, batch={batch_size})")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8)
                path = output_dir / "figures" / f"transformer_step_memory_{model_size}_{precision}.png"
                savefig(path)
                paths[f"transformer_step_memory_{model_size}_{precision}"] = path

    return paths


def write_analysis(
    output_dir: Path,
    attention: pd.DataFrame,
    memory: pd.DataFrame,
    transformer: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    if not attention.empty:
        summary_rows.append({"domain": "attention", "rows": len(attention), "runs": attention["run_id"].nunique()})
    if not memory.empty:
        summary_rows.append({"domain": "memory-accounting", "rows": len(memory), "runs": memory["run_id"].nunique()})
    if not transformer.empty:
        summary_rows.append({"domain": "transformer-step", "rows": len(transformer), "runs": transformer["run_id"].nunique()})
    write_markdown_table(summary_rows, output_dir / "summary.md")

    lines = [
        "# Experiment Analysis Index",
        "",
        "## Included Data",
        "",
    ]
    for row in summary_rows:
        lines.append(f"- {row['domain']}: {row['rows']} rows from {row['runs']} runs")

    lines.extend(["", "## Figures", ""])
    for name, path in sorted(figure_paths.items()):
        rel = path.relative_to(output_dir)
        lines.append(f"- {name}: [{rel}]({rel})")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Figures are generated from `results.csv` files only.",
            "- Attention backward numbers for `flash_triton` reflect the current project implementation, whose backward path uses compiled PyTorch.",
            "- Distributed memory accounting is theoretical persistent memory and excludes activations and transient communication buffers.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    attention = read_domain_results(args.results_root, "attention")
    memory = read_domain_results(args.results_root, "memory-accounting")
    transformer = read_domain_results(args.results_root, "transformer-step")

    if not attention.empty:
        attention.to_csv(output_dir / "all_attention_results.csv", index=False)
    if not memory.empty:
        memory.to_csv(output_dir / "all_memory_accounting_results.csv", index=False)
    if not transformer.empty:
        transformer.to_csv(output_dir / "all_transformer_step_results.csv", index=False)

    figure_paths: dict[str, Path] = {}
    figure_paths.update(plot_attention(attention, output_dir))
    figure_paths.update(plot_memory_accounting(memory, output_dir))
    figure_paths.update(plot_transformer_step(transformer, output_dir))
    write_analysis(output_dir, attention, memory, transformer, figure_paths)
    print(f"Saved analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
