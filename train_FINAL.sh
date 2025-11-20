#!/bin/bash
set -e

echo "=========================================="
echo "🚀 TCM模型训练 - 最终版"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH=$(find ~/.cache/modelscope/hub/ -type d -path "*/Qwen/Qwen2___5-7B-Instruct" | grep -v temp | head -1)

echo ""
echo "配置检查:"
echo "  GPU: 4-7"
echo "  模型: $MODEL_PATH"
echo "  数据集: tcm_train_30p"
echo ""

mkdir -p log models/checkpoints

LOG="log/train_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志: $LOG"
echo ""
echo "🚀 开始训练..."
echo ""

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset tcm_train_30p \
    --dataset_dir ./config \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 64 \
    --lora_alpha 16 \
    --output_dir ./models/checkpoints/qwen2.5-7b-tcm-lora \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 \
    --gradient_checkpointing \
    --quantization_bit 4 \
    --preprocessing_num_workers 16 \
    2>&1 | tee -a "$LOG"

echo ""
echo "✅ 训练完成！日志: $LOG"
