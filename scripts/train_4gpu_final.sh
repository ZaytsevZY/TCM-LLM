#!/bin/bash
set -e

echo "=========================================="
echo "🚀 TCM模型训练 - 4卡版（GPU 4-7）"
echo "=========================================="
echo ""

# 使用GPU 4-7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 验证
echo "检查环境..."
python << 'PYTHON'
import json
import os

# 检查配置
with open('config/dataset_info.json', 'r') as f:
    config = json.load(f)
    
if 'tcm_train_50p' not in config:
    raise ValueError("配置中缺少 tcm_train_50p")

# 检查数据文件
data_file = f"data/jsonl/{config['tcm_train_50p']['file_name']}"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"找不到数据文件: {data_file}")

with open(data_file, 'r') as f:
    count = sum(1 for _ in f)
    
print(f"✓ 配置验证通过")
print(f"✓ 数据文件: {data_file}")
print(f"✓ 数据量: {count:,} 条")
PYTHON

echo ""

# 获取模型路径
MODEL_PATH=$(find ~/.cache/modelscope/hub/ -type d -path "*/Qwen/Qwen2___5-7B-Instruct" | grep -v temp | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "❌ 找不到模型"
    exit 1
fi

echo "✓ 模型路径: $MODEL_PATH"
echo ""

# 创建目录
mkdir -p log models/checkpoints

# 日志文件
LOG_FILE="log/train_4gpu_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志: $LOG_FILE"
echo ""
echo "🚀 开始训练（预计18-20小时）..."
echo ""

# 训练
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_PATH" \
    --dataset tcm_train_50p \
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
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ 训练完成"
echo "=========================================="
echo "日志: $LOG_FILE"
