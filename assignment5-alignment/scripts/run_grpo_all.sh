#!/bin/bash

# ================= 配置区 =================
BASE_MODEL="model/Qwen2.5-Math-1.5B"

TRAIN_DATA="data/math12k/data/train-00000-of-00001.parquet"
TEST_DATA="data/math12k/data/test-00000-of-00001.parquet"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"

OUTPUT_BASE="result/grpo_4loss_ablation_lr_scaled"
WANDB_PROJECT="cs336-grpo-math12k-after-base-4loss-lr-scaled"

# ================= 超参数 =================
ANCHOR_LR=3e-5
ANCHOR_BATCH=256

N_STEPS=200
ROLLOUT_SIZE=256
GROUP_SIZE=8
TRAIN_BATCH=256
MICRO_BS=2
ACCUM_STEPS=$((TRAIN_BATCH / MICRO_BS))
EPOCHS_PER_ROLLOUT=1

# 实际学习率 = Anchor_LR * (train_batch_size / Anchor_Batch) / sqrt(epoch)
BEST_LR=$(python -c "import math; print(f'{$ANCHOR_LR * ($TRAIN_BATCH / $ANCHOR_BATCH) / math.sqrt($EPOCHS_PER_ROLLOUT):.8g}')")

LENGTH_NORM_TYPE="mask_normalize"
STD_FLAG="--use_std_normalization"
STD_DESC="enabled"

# ================= 实验矩阵 =================
EXPERIMENTS=(
    "no_baseline:no_baseline"
    "grpo_baseline:reinforce_with_baseline"
    "grpo_no_clip:grpo_no_clip"
    "grpo_clip:grpo_clip"
)

echo "======================================================="
echo "📌 Anchor LR: $ANCHOR_LR"
echo "📌 Anchor Batch: $ANCHOR_BATCH"
echo "📌 Train Batch: $TRAIN_BATCH"
echo "📌 Epochs per Rollout: $EPOCHS_PER_ROLLOUT"
echo "📌 Scaled LR: $BEST_LR"
echo "======================================================="

for EXP in "${EXPERIMENTS[@]}"; do
    NAME_SUFFIX=${EXP%%:*}
    LOSS_TYPE=${EXP#*:}

    RUN_NAME="E${EPOCHS_PER_ROLLOUT}_TB${TRAIN_BATCH}_LR${BEST_LR}_${NAME_SUFFIX}"
    EXP_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"

    echo "======================================================="
    echo "🚀 启动 4-loss 消融实验: $RUN_NAME"
    echo "📌 loss_type: $LOSS_TYPE"
    echo "📏 length_norm_type: $LENGTH_NORM_TYPE"
    echo "📊 std_normalization: $STD_DESC"
    echo "📊 scaled_lr: $BEST_LR"
    echo "📂 output_dir: $EXP_OUTPUT_DIR"
    echo "======================================================="

    python -m cs336_alignment.train_grpo \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --test_data_path "$TEST_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "$EXP_OUTPUT_DIR" \
        $STD_FLAG \
        --length_norm_type "$LENGTH_NORM_TYPE" \
        --n_grpo_steps "$N_STEPS" \
        --lr "$BEST_LR" \
        --rollout_batch_size "$ROLLOUT_SIZE" \
        --group_size "$GROUP_SIZE" \
        --train_batch_size "$TRAIN_BATCH" \
        --gradient_accumulation_steps "$ACCUM_STEPS" \
        --epochs_per_rollout_batch "$EPOCHS_PER_ROLLOUT" \
        --loss_type "$LOSS_TYPE" \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.8 \
        --eval_every_steps 8 \
        --save_every_steps "$N_STEPS" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME" \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 失败！"
        exit 1
    fi

    echo "✅ 实验 $RUN_NAME 完成！"
    echo "-------------------------------------------------------"
    sleep 10
done

echo "🎉 所有 4-loss 消融实验执行完毕！"