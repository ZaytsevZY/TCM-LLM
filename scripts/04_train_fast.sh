cd ~/TCM-LLM

# ========== 停止所有 ==========
pkill -9 -f llamafactory 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
sleep 2

# ========== 使用已有的30%数据 ==========
# 如果已经创建了 train_30p.jsonl 就用它，否则创建
if [ ! -f data/jsonl/train_30p.jsonl ]; then
    python << 'PY'
import random
random.seed(42)
with open('data/jsonl/train.jsonl', 'r') as f:
    lines = f.readlines()
sampled = random.sample(lines, len(lines) // 3)
with open('data/jsonl/train_30p.jsonl', 'w') as f:
    f.writelines(sampled)
print(f"✓ 创建30%数据: {len(sampled):,} 条")
PY
fi

# ========== 修复配置文件 ==========
cat > config/dataset_info.json << 'EOF'
{
  "tcm_train": {
    "file_name": "../data/jsonl/train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  },
  "tcm_train_30p": {
    "file_name": "../data/jsonl/train_30p.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
EOF

# ========== 创建最终训练脚本 ==========
cat > train_FINAL.sh << 'SCRIPT'
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
SCRIPT

chmod +x train_FINAL.sh

# ========== 验证 ==========
echo ""
echo "=========================================="
echo "验证配置"
echo "=========================================="

python << 'PY'
import json
import os

# 检查配置
with open('config/dataset_info.json', 'r') as f:
    cfg = json.load(f)

print("数据集配置:")
for name, info in cfg.items():
    path = os.path.join('config', info['file_name'])
    exists = os.path.exists(path)
    if exists:
        count = sum(1 for _ in open(path))
        print(f"  ✓ {name}: {count:,} 条")
    else:
        print(f"  ✗ {name}: 文件不存在")

# 检查是否有tcm_train_30p
if 'tcm_train_30p' not in cfg:
    print("\n⚠️  配置中缺少 tcm_train_30p")
else:
    print("\n✓ tcm_train_30p 配置正确")
PY

echo ""
echo "=========================================="
echo "✅ 准备完成！"
echo ""
echo "现在运行:"
echo "  bash train_FINAL.sh"
echo "=========================================="