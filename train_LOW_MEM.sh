#!/bin/bash
set -e

echo "=========================================="
echo "🚀 TCM模型训练 - 低显存优化版"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH=$(find ~/.cache/modelscope/hub/ -type d -path "*/Qwen/Qwen2___5-7B-Instruct" | grep -v temp | head -1)

echo ""
echo "低显存优化配置:"
echo "  Batch size: 2 (降低)"
echo "  Gradient accumulation: 8 (增加以保持总batch=64)"
echo "  LoRA rank: 32 (减半)"
echo "  Sequence length: 1024 (减半)"
echo ""

mkdir -p log models/checkpoints

LOG="log/train_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志: $LOG"
echo ""
echo "�� 开始训练..."
echo ""

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset tcm_train_30p \
    --dataset_dir ./config \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 32 \
    --lora_alpha 64 \
    --output_dir ./models/checkpoints/qwen2.5-7b-tcm-lora \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 1000 \
    --bf16 \
    --gradient_checkpointing \
    --quantization_bit 4 \
    --preprocessing_num_workers 16 \
    2>&1 | tee -a "$LOG"

echo ""
echo "✅ 训练完成！日志: $LOG"
