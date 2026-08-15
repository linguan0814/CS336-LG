#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[flash-attention] focused 8GB benchmark"
echo "[flash-attention] project=${PROJECT_ROOT}"

echo "[flash-attention] correctness gate"
uv run pytest tests/test_attention.py -q

echo "[flash-attention] non-causal long-sequence benchmark"
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cuda \
  --batch-size 1 \
  --seq-lengths 1024 2048 4096 8192 \
  --d-models 64 128 \
  --implementations torch torch_compile flash_triton \
  --warmup-steps 3 \
  --timing-steps 10 \
  --run-label focused-8gb-noncausal

echo "[flash-attention] causal long-sequence benchmark"
uv run python scripts/experiments/benchmark_flash_attention.py \
  --device cuda \
  --batch-size 1 \
  --seq-lengths 1024 2048 4096 8192 \
  --d-models 64 128 \
  --implementations torch torch_compile flash_triton \
  --causal \
  --warmup-steps 3 \
  --timing-steps 10 \
  --run-label focused-8gb-causal

echo "[flash-attention] done"
