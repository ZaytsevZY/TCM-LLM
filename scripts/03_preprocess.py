#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 (修正版)
功能：正确处理instruction+input格式，划分数据集
"""
import os
import json
import random
from datasets import load_from_disk
from tqdm import tqdm

print("=" * 60)
print("🔧 中医诊疗系统 - 数据预处理 (修正版)")
print("=" * 60)

# 设置随机种子
random.seed(42)

# 创建输出目录
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/evaluation", exist_ok=True)

# ==========================================
# 1. 加载原始数据
# ==========================================
print("\n[1/6] 📂 加载原始数据...")
try:
    tcm_sft = load_from_disk("data/raw/tcm_sft")
    total_size = len(tcm_sft['train'])
    print(f"✓ 加载成功: {total_size:,} 条数据")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    exit(1)

# ==========================================
# 2. 分析数据格式
# ==========================================
print("\n[2/6] 🔍 分析数据格式...")

# 检查前100条数据
sample_size = min(100, len(tcm_sft['train']))
samples = tcm_sft['train'].select(range(sample_size))

# 统计input字段的使用情况
empty_input_count = 0
non_empty_input_count = 0
input_lengths = []

for item in samples:
    if not item.get('input', '').strip():
        empty_input_count += 1
    else:
        non_empty_input_count += 1
        input_lengths.append(len(item['input']))

print(f"数据格式分析（前{sample_size}条）:")
print(f"  instruction字段: {sample_size}/{sample_size} 有内容")
print(f"  input字段为空: {empty_input_count} 条 ({empty_input_count/sample_size*100:.1f}%)")
print(f"  input字段有内容: {non_empty_input_count} 条 ({non_empty_input_count/sample_size*100:.1f}%)")
if input_lengths:
    print(f"  input平均长度: {sum(input_lengths)/len(input_lengths):.0f} 字符")

# 显示样例
print(f"\n样例1（input为空）:")
for item in samples:
    if not item.get('input', '').strip():
        print(f"  instruction: {item['instruction'][:80]}...")
        print(f"  input: (空)")
        print(f"  output: {item['output'][:80]}...")
        break

print(f"\n样例2（input有内容）:")
for item in samples:
    if item.get('input', '').strip():
        print(f"  instruction: {item['instruction'][:80]}...")
        print(f"  input: {item['input'][:80]}...")
        print(f"  output: {item['output'][:80]}...")
        break

# ==========================================
# 3. 定义数据格式化函数
# ==========================================
def format_question(item):
    """
    格式化完整的问题
    如果input为空，只用instruction
    如果input不为空，组合instruction和input
    """
    instruction = item['instruction'].strip()
    input_text = item.get('input', '').strip()
    
    if input_text:
        # input有内容，组合两者
        return f"{instruction}\n\n补充信息：\n{input_text}"
    else:
        # input为空，只用instruction
        return instruction

def create_formatted_item(item, index):
    """创建格式化后的数据项"""
    return {
        "id": index,
        "instruction": item['instruction'],
        "input": item.get('input', ''),
        "output": item['output'],
        "full_question": format_question(item)  # 新增：完整问题
    }

# ==========================================
# 4. 数据划分
# ==========================================
print("\n[3/6] ✂️  划分数据集...")
print("划分比例: 训练集95% / 验证集2% / 测试集3%")

# 可选：快速测试模式
USE_FULL_DATA = True  # 改为False使用10万条快速测试
if not USE_FULL_DATA:
    print("⚠️  使用采样模式（10万条），适合快速测试")
    tcm_sft['train'] = tcm_sft['train'].shuffle(seed=42).select(range(100000))
    total_size = len(tcm_sft['train'])

# 第一次划分：训练集 vs 临时集
train_test_split_ratio = 0.05
print(f"\n步骤1: 分离训练集和临时集...")
split1 = tcm_sft['train'].train_test_split(test_size=train_test_split_ratio, seed=42)
train_dataset = split1['train']
temp_dataset = split1['test']

print(f"  训练集: {len(train_dataset):,} 条 (95%)")
print(f"  临时集: {len(temp_dataset):,} 条 (5%)")

# 第二次划分：验证集 vs 测试集
print(f"\n步骤2: 分离验证集和测试集...")
split2 = temp_dataset.train_test_split(test_size=0.6, seed=42)
val_dataset = split2['train']
test_dataset = split2['test']

print(f"  验证集: {len(val_dataset):,} 条 (2%)")
print(f"  测试集: {len(test_dataset):,} 条 (3%)")

# ==========================================
# 5. 保存处理后的数据
# ==========================================
print("\n[4/6] 💾 保存数据集...")

print("保存训练集...")
train_dataset.save_to_disk("data/processed/train")
print(f"  ✓ data/processed/train/")

print("保存验证集...")
val_dataset.save_to_disk("data/processed/val")
print(f"  ✓ data/processed/val/")

print("保存测试集...")
test_dataset.save_to_disk("data/processed/test")
print(f"  ✓ data/processed/test/")

# ==========================================
# 6. 准备评测数据集（带完整问题）
# ==========================================
print("\n[5/6] 🎯 准备评测数据集...")

# 快速评测集：100条
print("创建快速评测集（100条）...")
eval_100 = test_dataset.shuffle(seed=42).select(range(min(100, len(test_dataset))))
eval_100_list = []
for idx, item in enumerate(eval_100):
    formatted_item = create_formatted_item(item, idx)
    eval_100_list.append(formatted_item)

with open("data/evaluation/eval_100.json", "w", encoding="utf-8") as f:
    json.dump(eval_100_list, f, ensure_ascii=False, indent=2)
print(f"  ✓ data/evaluation/eval_100.json")

# 完整评测集：500条
print("创建完整评测集（500条）...")
eval_500 = test_dataset.shuffle(seed=42).select(range(min(500, len(test_dataset))))
eval_500_list = []
for idx, item in enumerate(eval_500):
    formatted_item = create_formatted_item(item, idx)
    eval_500_list.append(formatted_item)

with open("data/evaluation/eval_500.json", "w", encoding="utf-8") as f:
    json.dump(eval_500_list, f, ensure_ascii=False, indent=2)
print(f"  ✓ data/evaluation/eval_500.json")

# 保存格式说明文档
format_doc = """
# 评测数据格式说明

每条数据包含以下字段：

1. id: 数据编号
2. instruction: 主要问题/指令
3. input: 补充信息（可能为空）
4. output: 标准答案
5. full_question: 完整问题（instruction + input组合）

## 使用方法

### 零样本推理
使用 full_question 作为输入，直接提问模型。

### CoT推理
将 full_question 嵌入CoT prompt模板中。

## 注意事项

- 如果input为空，full_question = instruction
- 如果input不为空，full_question = instruction + "\\n\\n补充信息：\\n" + input
"""

with open("data/evaluation/FORMAT.md", "w", encoding="utf-8") as f:
    f.write(format_doc)
print(f"  ✓ data/evaluation/FORMAT.md (格式说明)")

# ==========================================
# 7. 数据统计
# ==========================================
print("\n[6/6] 📊 数据统计...")

def get_stats(dataset, sample_size=1000):
    """计算数据集统计信息"""
    sample_size = min(sample_size, len(dataset))
    sample = dataset.select(range(sample_size))
    
    stats = {
        'empty_input': 0,
        'non_empty_input': 0,
        'output_lengths': [],
        'input_lengths': []
    }
    
    for item in sample:
        input_text = item.get('input', '').strip()
        if not input_text:
            stats['empty_input'] += 1
        else:
            stats['non_empty_input'] += 1
            stats['input_lengths'].append(len(input_text))
        
        stats['output_lengths'].append(len(item['output']))
    
    return stats

train_stats = get_stats(train_dataset)
val_stats = get_stats(val_dataset)
test_stats = get_stats(test_dataset)

print("\ninput字段统计:")
print(f"  训练集: input为空={train_stats['empty_input']}, 有内容={train_stats['non_empty_input']}")
print(f"  验证集: input为空={val_stats['empty_input']}, 有内容={val_stats['non_empty_input']}")
print(f"  测试集: input为空={test_stats['empty_input']}, 有内容={test_stats['non_empty_input']}")

print("\noutput长度统计:")
train_avg = sum(train_stats['output_lengths']) / len(train_stats['output_lengths'])
print(f"  训练集: 平均={train_avg:.0f}, 最小={min(train_stats['output_lengths'])}, 最大={max(train_stats['output_lengths'])}")

print("\n数据集规模:")
print(f"  训练集: {len(train_dataset):,} 条 (95%)")
print(f"  验证集: {len(val_dataset):,} 条 (2%)")
print(f"  测试集: {len(test_dataset):,} 条 (3%)")
print(f"  快速评测: 100 条")
print(f"  完整评测: 500 条")

# ==========================================
# 完成
# ==========================================
print("\n" + "=" * 60)
print("✅ 数据预处理完成！")
print("=" * 60)

print("\n📁 输出文件:")
print("  data/processed/train/          - 训练集")
print("  data/processed/val/            - 验证集")
print("  data/processed/test/           - 测试集")
print("  data/evaluation/eval_100.json  - 快速评测集（带full_question）")
print("  data/evaluation/eval_500.json  - 完整评测集（带full_question）")
print("  data/evaluation/FORMAT.md      - 数据格式说明")

print("\n🎯 下一步:")
print("  1. 检查数据: python scripts/check_data_quality.py")
print("  2. 配置训练: 编辑 config/model_config.yaml")
print("  3. 开始训练: bash scripts/04_train.sh")
