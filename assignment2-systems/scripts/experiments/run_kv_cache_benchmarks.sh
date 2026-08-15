#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[kv-cache] focused 8GB benchmark"
echo "[kv-cache] project=${PROJECT_ROOT}"

echo "[kv-cache] correctness gate"
uv run pytest tests/test_kv_cache.py -q

echo "[kv-cache] 8GB xlarge benchmark"
uv run python scripts/experiments/benchmark_kv_cache.py \
  --device cuda \
  --batch-size 1 \
  --prompt-lengths 512 1024 \
  --max-new-tokens 128 256 \
  --vocab-size 16384 \
  --context-length 1536 \
  --d-model 768 \
  --num-layers 12 \
  --num-heads 12 \
  --d-ff 3072 \
  --warmup-steps 3 \
  --timing-steps 8 \
  --run-label kv-cache-8gb-xlarge

echo "[kv-cache] 8GB pressure benchmark"
uv run python scripts/experiments/benchmark_kv_cache.py \
  --device cuda \
  --batch-size 1 \
  --prompt-lengths 1024 1536 \
  --max-new-tokens 128 256 \
  --vocab-size 16384 \
  --context-length 2048 \
  --d-model 768 \
  --num-layers 16 \
  --num-heads 12 \
  --d-ff 3072 \
  --warmup-steps 3 \
  --timing-steps 6 \
  --run-label kv-cache-8gb-pressure

echo "[kv-cache] done"
