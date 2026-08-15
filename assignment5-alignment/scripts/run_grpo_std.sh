#!/bin/bash

# ================= 配置区 =================
BASE_MODEL="model/Qwen2.5-Math-1.5B" # base model
# TRAIN_DATA="data/gsm8k/train.jsonl"
# TEST_DATA="data/gsm8k/test.jsonl"

TRAIN_DATA="data/math12k/data/train-00000-of-00001.parquet"
TEST_DATA="data/math12k/data/test-00000-of-00001.parquet"


PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/grpo_std"
WANDB_PROJECT="cs336-grpo-math12k-after-base-std"

BEST_LR=3e-5 
N_STEPS=200
ROLLOUT_SIZE=256
GROUP_SIZE=8
TRAIN_BATCH=256
MICRO_BS=2
ACCUM_STEPS=$((TRAIN_BATCH / MICRO_BS))
EPOCHS_PER_ROLLOUT=1

# ================= 定义实验矩阵 =================
EXPERIMENTS=(
    "mask_normalize:with_std"
    "mask_normalize:no_std"
)

for EXP in "${EXPERIMENTS[@]}"; do
    # 解析参数
    NORM_TYPE=${EXP%%:*}
    STD_TYPE=${EXP#*:}

    # 设置命令参数
    if [ "$STD_TYPE" == "with_std" ]; then
        STD_FLAG="--use_std_normalization"
    else
        STD_FLAG=""
    fi

    # 生成运行名称
    RUN_NAME="grpo_${NORM_TYPE}_${STD_TYPE}_lr${BEST_LR}"
    EXP_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"

    echo "======================================================="
    echo "🚀 启动高级消融实验: $RUN_NAME"
    echo "📏 长度归一化: $NORM_TYPE"
    echo "📊 标准差归一化: $STD_TYPE"
    echo "======================================================="

    python -m cs336_alignment.train_grpo \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --test_data_path "$TEST_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "$EXP_OUTPUT_DIR" \
        $STD_FLAG \
        --length_norm_type "$NORM_TYPE" \
        --n_grpo_steps "$N_STEPS" \
        --lr "$BEST_LR" \
        --rollout_batch_size "$ROLLOUT_SIZE" \
        --group_size "$GROUP_SIZE" \
        --train_batch_size "$TRAIN_BATCH" \
        --gradient_accumulation_steps "$ACCUM_STEPS" \
        --epochs_per_rollout_batch "$EPOCHS_PER_ROLLOUT" \
        --loss_type "grpo_clip" \
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

echo "🎉 所有高级消融实验执行完毕！"
