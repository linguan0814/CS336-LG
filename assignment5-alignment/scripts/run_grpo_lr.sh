#!/bin/bash
# export CUDA_VISIBLE_DEVICES='1,2'
# uv run bash cs336_alignment/run_grpo_lr.sh
# ================= 配置区 =================

BASE_MODEL="model/Qwen2.5-Math-1.5B" # base model
# BASE_MODEL="result/checkpoints/sft_subset7395_filteredTrue"  # sft model

# 数据与模版
# TRAIN_DATA="data/gsm8k/train.jsonl"
# TEST_DATA="data/gsm8k/test.jsonl"
TRAIN_DATA="data/math12k/data/train-00000-of-00001.parquet"
TEST_DATA="data/math12k/data/test-00000-of-00001.parquet"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/grpo_lr_sweep"

# WANDB_PROJECT="cs336-grpo-after-base-lr-grpo_clip"
WANDB_PROJECT="cs336-grpo-math12k-after-base-lr-grpo_clip"
# 待测试的学习率列
LR_LIST=(3e-6 5e-6 1e-5 3e-5 5e-5)
N_STEPS=200
ROLLOUT_SIZE=256
GROUP_SIZE=8
TRAIN_BATCH=256
MICRO_BS=2
ACCUM_STEPS=$((TRAIN_BATCH / MICRO_BS))
EPOCHS_PER_ROLLOUT=1
# ================= 循环运行 =================
for LR in "${LR_LIST[@]}"; do
    RUN_NAME="grpo_lr${LR}"
    echo "======================================================="
    echo "🚀 [LR Sweep] 启动实验: $RUN_NAME (LR=$LR)"
    echo "======================================================="
    
    python -m cs336_alignment.train_grpo \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --test_data_path "$TEST_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "${OUTPUT_BASE}/${RUN_NAME}" \
        --n_grpo_steps "$N_STEPS" \
        --lr "$LR" \
        --rollout_batch_size "$ROLLOUT_SIZE" \
        --group_size "$GROUP_SIZE" \
        --train_batch_size "$TRAIN_BATCH" \
        --gradient_accumulation_steps "$ACCUM_STEPS" \
        --epochs_per_rollout_batch "$EPOCHS_PER_ROLLOUT" \
        --loss_type "grpo_clip" \
        --eval_every_steps 8 \
        --save_every_steps "$N_STEPS" \
        --use_std_normalization \
        --length_norm_type "mask_normalize" \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.8 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME" \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 失败，跳过..."
    else
        echo "✅ 实验 $RUN_NAME 完成！"
    fi
    
    # 稍微休息释放显存
    sleep 5
done
