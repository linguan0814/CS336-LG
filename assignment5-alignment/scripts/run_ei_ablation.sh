#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 两张 GPU：cuda:0 训练 policy，cuda:1 跑 vLLM。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_PROJECT="${WANDB_PROJECT:-cs336-ei-ablation}"

MODEL_ID="${MODEL_ID:-result/checkpoints/sft_steps200_subsetNone_filteredFalse}"
TRAIN_DATA="${TRAIN_DATA:-data/gsm8k/train.jsonl}"
VAL_DATA="${VAL_DATA:-data/gsm8k/test.jsonl}"
PROMPT_PATH="${PROMPT_PATH:-cs336_alignment/prompts/r1_zero.prompt}"
N_EI_STEPS="${N_EI_STEPS:-5}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# 实验设计：
# - 从同一个 SFT checkpoint 出发。
# - 每轮 rollout budget 固定为 4096 条生成，只改变 D_b 和 G 的分配。
# - 预期 EI 对验证集 accuracy 提升不大，因此用较大的验证集减少噪声。
# - 服务器剩余磁盘少：关闭 checkpoint 保存，只看 W&B 指标。
COMMON_ARGS=(
  --model_id "$MODEL_ID"
  --train_data_path "$TRAIN_DATA"
  --val_data_path "$VAL_DATA"
  --prompt_path "$PROMPT_PATH"
  --output_dir /tmp/cs336_ei_no_save
  --lr 1e-5
  --batch_size 16
  --micro_batch_size 1
  --n_ei_steps "$N_EI_STEPS"
  --epochs_per_ei 1
  --max_tokens "$MAX_TOKENS"
  --eval_every_steps 100000
  --max_eval_samples "$MAX_EVAL_SAMPLES"
  --device cuda:0
  --vllm_device cuda:1
  --vllm_gpu_util 0.2
  --wandb_project "$WANDB_PROJECT"
  --no_save_checkpoints
)

run_ei() {
  local name="$1"
  local ei_batch_size="$2"
  local rollouts="$3"

  echo "===== EI ablation: ${name} | ei_batch_size=${ei_batch_size}, rollouts=${rollouts} ====="
  python -m cs336_alignment.train_ei_step \
    "${COMMON_ARGS[@]}" \
    --ei_batch_size "$ei_batch_size" \
    --rollouts "$rollouts" \
    --wandb_run_name "$name"
}

# Fixed per-iteration rollout budget: D_b * G = 4096.
# Compare whether EI benefits more from covering more unique questions or deeper sampling per question.
run_ei "ei_budget4096_db512_g8" 512 8
run_ei "ei_budget4096_db1024_g4" 1024 4
run_ei "ei_budget4096_db2048_g2" 2048 2
