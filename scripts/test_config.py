#!/usr/bin/env python3
"""测试训练配置是否正确"""
import os
import json
from datasets import load_from_disk

print("🔍 测试训练配置...\n")

# 1. 检查数据集配置
print("[1] 检查dataset_info.json...")
with open("config/dataset_info.json", 'r') as f:
    dataset_info = json.load(f)
    print(f"  ✓ 配置文件读取成功")
    print(f"  ✓ 训练集配置: {list(dataset_info.keys())}")

# 2. 检查数据集是否存在
print("\n[2] 检查数据集文件...")
for name, info in dataset_info.items():
    path = info['file_name'].replace('../', '')
    if os.path.exists(path):
        ds = load_from_disk(path)
        print(f"  ✓ {name}: {len(ds):,} 条")
    else:
        print(f"  ✗ {name}: 未找到 ({path})")

# 3. 检查数据格式
print("\n[3] 检查数据格式...")
train = load_from_disk("data/processed/train")
sample = train[0]
required = ['instruction', 'input', 'output']
has_all = all(field in sample for field in required)
print(f"  字段: {list(sample.keys())}")
print(f"  格式检查: {'✓ 正确' if has_all else '✗ 缺少字段'}")

# 4. 检查目录
print("\n[4] 检查输出目录...")
dirs = ['models/checkpoints', 'outputs/logs']
for d in dirs:
    if os.path.exists(d):
        print(f"  ✓ {d}")
    else:
        os.makedirs(d)
        print(f"  ✓ {d} (已创建)")

print("\n" + "="*60)
print("✅ 配置检查完成！可以开始训练。")
print("\n下一步:")
print("  bash scripts/04_train.sh")
