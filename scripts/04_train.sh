#!/bin/bash
set -e

echo "=========================================="
echo "🚀 开始训练中医模型"
echo "=========================================="
echo ""
echo "GPU配置: 4 × 24GB (GPU 4-7)"
echo "训练数据: 3,493,840 条 (JSONL格式)"
echo "预计时间: 24-36小时"
echo ""

# 设置使用后四张GPU (4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 检查数据文件
if [ ! -f "data/jsonl/train.jsonl" ]; then
    echo "❌ 错误: 找不到 data/jsonl/train.jsonl"
    echo "请先运行: python scripts/convert_to_jsonl.py"
    exit 1
fi

echo "✓ 数据文件检查通过"
echo ""

# 获取模型路径
MODEL_PATH=$(find ~/.cache/modelscope/hub/ -type d -path "*/Qwen/Qwen2___5-7B-Instruct" | grep -v temp | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "❌ 错误: 找不到 ModelScope 下载的模型"
    echo "请先运行: bash scripts/download_model_modelscope.sh"
    exit 1
fi

echo "✓ 找到模型: $MODEL_PATH"
echo ""

# 创建日志目录
mkdir -p log models/checkpoints

# 生成日志文件名
LOG_FILE="log/train_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件: $LOG_FILE"
echo ""

# 启动训练（使用 tee 同时输出到终端和文件，-a 追加模式）
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset tcm_train \
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
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --bf16 \
    --gradient_checkpointing \
    --quantization_bit 4 \
    --preprocessing_num_workers 8 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo "📝 完整日志已保存至: $LOG_FILE"
