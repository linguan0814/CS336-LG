#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[a2] focused 8GB benchmark suite"
echo "[a2] this run skips tiny smoke benchmarks and conservative small-model sweeps"

echo "[a2] running FlashAttention focused benchmarks"
"${SCRIPT_DIR}/run_flash_attention_benchmarks.sh"

echo "[a2] running KV Cache focused benchmarks"
"${SCRIPT_DIR}/run_kv_cache_benchmarks.sh"

echo "[a2] all focused benchmarks finished"
