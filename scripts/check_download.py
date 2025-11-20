#!/usr/bin/env python3
"""详细检查数据质量"""
import os
from datasets import load_from_disk

print("🔍 详细检查数据...")

if os.path.exists("data/raw/tcm_sft"):
    tcm = load_from_disk("data/raw/tcm_sft")
    
    # 检查前10条数据
    print("\n📊 前10条数据检查:")
    for i in range(min(10, len(tcm['train']))):
        sample = tcm['train'][i]
        inst_len = len(sample.get('instruction', ''))
        inp_len = len(sample.get('input', ''))
        out_len = len(sample.get('output', ''))
        
        print(f"\n样例 {i+1}:")
        print(f"  instruction长度: {inst_len}")
        print(f"  input长度: {inp_len}")
        print(f"  output长度: {out_len}")
        
        if inst_len > 0:
            print(f"  instruction: {sample['instruction'][:80]}...")
        if inp_len > 0:
            print(f"  input: {sample['input'][:80]}...")
        if out_len > 0:
            print(f"  output: {sample['output'][:80]}...")
    
    # 统计空字段
    print("\n📈 数据统计:")
    empty_inst = sum(1 for x in tcm['train'] if not x.get('instruction', '').strip())
    empty_input = sum(1 for x in tcm['train'] if not x.get('input', '').strip())
    empty_output = sum(1 for x in tcm['train'] if not x.get('output', '').strip())
    
    total = len(tcm['train'])
    print(f"  总数据: {total:,}")
    print(f"  空instruction: {empty_inst:,} ({empty_inst/total*100:.1f}%)")
    print(f"  空input: {empty_input:,} ({empty_input/total*100:.1f}%)")
    print(f"  空output: {empty_output:,} ({empty_output/total*100:.1f}%)")
    
else:
    print("✗ 数据未找到")