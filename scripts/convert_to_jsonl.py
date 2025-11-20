#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Arrow格式转换为JSONL格式
LLamaFactory更好地支持JSONL
"""
import os
import json
from datasets import load_from_disk
from tqdm import tqdm

print("=" * 60)
print("🔄 转换数据格式: Arrow → JSONL")
print("=" * 60)

# 创建JSONL目录
os.makedirs("data/jsonl", exist_ok=True)

# 1. 转换训练集
print("\n[1/2] 转换训练集...")
train = load_from_disk("data/processed/train")
print(f"加载数据: {len(train):,} 条")

with open("data/jsonl/train.jsonl", 'w', encoding='utf-8') as f:
    for item in tqdm(train, desc="写入训练集"):
        # 格式化为LLamaFactory期望的格式
        record = {
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"✓ 训练集已保存: data/jsonl/train.jsonl")

# 2. 转换验证集
print("\n[2/2] 转换验证集...")
val = load_from_disk("data/processed/val")
print(f"加载数据: {len(val):,} 条")

with open("data/jsonl/val.jsonl", 'w', encoding='utf-8') as f:
    for item in tqdm(val, desc="写入验证集"):
        record = {
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"✓ 验证集已保存: data/jsonl/val.jsonl")

# 验证文件
print("\n" + "=" * 60)
print("📊 文件统计:")
import subprocess
train_size = subprocess.check_output(['wc', '-l', 'data/jsonl/train.jsonl']).decode().split()[0]
val_size = subprocess.check_output(['wc', '-l', 'data/jsonl/val.jsonl']).decode().split()[0]
print(f"  train.jsonl: {train_size} 行")
print(f"  val.jsonl: {val_size} 行")

# 显示样例
print("\n样例数据 (train.jsonl):")
with open("data/jsonl/train.jsonl", 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())
    print(f"  instruction: {sample['instruction'][:80]}...")
    print(f"  input: {sample['input'][:80] if sample['input'] else '(空)'}...")
    print(f"  output: {sample['output'][:80]}...")

print("\n" + "=" * 60)
print("✅ 转换完成！")
print("=" * 60)
