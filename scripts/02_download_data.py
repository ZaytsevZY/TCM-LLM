#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集下载脚本
下载中医SFT数据集和可选的COIG数据集
"""
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse

print("=" * 60)
print("📥 中医诊疗系统 - 数据集下载")
print("=" * 60)

# 创建数据目录
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/raw/cache", exist_ok=True)

# 解析命令行参数
parser = argparse.ArgumentParser(description='下载数据集')
parser.add_argument('--skip-coig', action='store_true', 
                    help='跳过COIG数据集下载（节省时间）')
args = parser.parse_args()

# ==========================================
# 1. 下载主数据集：中医SFT数据
# ==========================================
print("\n[1/2] 📚 下载中医SFT数据集...")
print("数据集: SylvanL/Traditional-Chinese-Medicine-Dataset-SFT")
print("规模: 3,677,727 条")
print("预计时间: 30-60分钟（取决于网络速度）")
print("-" * 60)

try:
    print("正在连接HuggingFace...")
    tcm_sft = load_dataset(
        "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT",
        cache_dir="./data/raw/cache",
        trust_remote_code=True
    )
    
    print(f"✓ 下载完成！")
    print(f"  训练集: {len(tcm_sft['train']):,} 条")
    
    # 保存到磁盘
    print("正在保存到磁盘...")
    tcm_sft.save_to_disk("./data/raw/tcm_sft")
    print(f"✓ 保存完成: data/raw/tcm_sft/")
    
    # 显示样例
    print("\n📝 数据样例:")
    sample = tcm_sft['train'][0]
    print(f"  问题: {sample['instruction'][:80]}...")
    print(f"  回答: {sample['output'][:80]}...")
    
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print("\n可能的原因:")
    print("  1. 网络连接问题（需要能访问HuggingFace）")
    print("  2. 磁盘空间不足（需要约2GB空间）")
    print("  3. 需要HuggingFace token")
    print("\n解决方案:")
    print("  - 配置代理: export HF_ENDPOINT=https://hf-mirror.com")
    print("  - 或使用VPN")
    exit(1)

# ==========================================
# 2. 下载辅助数据集：COIG-CQIA（可选）
# ==========================================
if not args.skip_coig:
    print("\n" + "=" * 60)
    print("[2/2] 📚 下载COIG-CQIA数据集（可选）...")
    print("用途: 保持通用中文能力，防止灾难性遗忘")
    print("如果时间紧张，可以使用 --skip-coig 跳过")
    print("-" * 60)
    
    try:
        coig = load_dataset(
            "m-a-p/COIG-CQIA",
            cache_dir="./data/raw/cache",
            trust_remote_code=True
        )
        print(f"✓ 下载完成: {len(coig['train']):,} 条")
        
        coig.save_to_disk("./data/raw/coig")
        print(f"✓ 保存完成: data/raw/coig/")
        
    except Exception as e:
        print(f"⚠ COIG下载失败（可跳过）: {e}")
        print("提示: 使用 --skip-coig 参数可以跳过此步骤")
else:
    print("\n[2/2] ⏭️  跳过COIG数据集下载")

# ==========================================
# 完成
# ==========================================
print("\n" + "=" * 60)
print("✅ 数据下载完成！")
print("=" * 60)

print("\n📊 已下载的数据集:")
if os.path.exists("data/raw/tcm_sft"):
    print("  ✓ 中医SFT: data/raw/tcm_sft/")
if os.path.exists("data/raw/coig"):
    print("  ✓ COIG-CQIA: data/raw/coig/")

print("\n🎯 下一步:")
print("  运行数据预处理: python scripts/03_preprocess.py")

