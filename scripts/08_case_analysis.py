#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细案例分析
对比零样本和CoT的实际回答质量
"""
import json
import sys

def analyze_cases(result_dir, num_cases=10):
    """分析典型案例"""
    
    # 加载4组结果
    with open(f"{result_dir}/api_baseline/zero_shot/predictions.json", 'r') as f:
        api_zs = json.load(f)
    with open(f"{result_dir}/api_baseline/cot/predictions.json", 'r') as f:
        api_cot = json.load(f)
    with open(f"{result_dir}/lora_finetuned/zero_shot/predictions.json", 'r') as f:
        lora_zs = json.load(f)
    with open(f"{result_dir}/lora_finetuned/cot/predictions.json", 'r') as f:
        lora_cot = json.load(f)
    
    print("=" * 80)
    print("📋 典型案例对比分析")
    print("=" * 80)
    
    for i in range(min(num_cases, len(api_zs))):
        print(f"\n{'='*80}")
        print(f"案例 {i+1}")
        print(f"{'='*80}")
        
        print(f"\n【问题】")
        print(f"{api_zs[i]['full_question'][:200]}...")
        
        print(f"\n【参考答案】({len(api_zs[i]['reference'])}字)")
        print(f"{api_zs[i]['reference'][:200]}...")
        
        print(f"\n【API零样本】({len(api_zs[i]['prediction'])}字)")
        print(f"{api_zs[i]['prediction'][:300]}...")
        
        print(f"\n【API-CoT】({len(api_cot[i]['prediction'])}字)")
        print(f"{api_cot[i]['prediction'][:300]}...")
        
        print(f"\n【LoRA零样本】({len(lora_zs[i]['prediction'])}字)")
        print(f"{lora_zs[i]['prediction'][:300]}...")
        
        print(f"\n【LoRA-CoT】({len(lora_cot[i]['prediction'])}字)")
        print(f"{lora_cot[i]['prediction'][:300]}...")
        
        print(f"\n{'='*80}")
        input("按Enter查看下一个案例...")


def length_analysis(result_dir):
    """分析回答长度"""
    
    with open(f"{result_dir}/api_baseline/zero_shot/predictions.json", 'r') as f:
        api_zs = json.load(f)
    with open(f"{result_dir}/api_baseline/cot/predictions.json", 'r') as f:
        api_cot = json.load(f)
    with open(f"{result_dir}/lora_finetuned/zero_shot/predictions.json", 'r') as f:
        lora_zs = json.load(f)
    with open(f"{result_dir}/lora_finetuned/cot/predictions.json", 'r') as f:
        lora_cot = json.load(f)
    
    print("\n" + "=" * 80)
    print("📏 回答长度分析")
    print("=" * 80)
    
    ref_len = sum(len(item['reference']) for item in api_zs) / len(api_zs)
    api_zs_len = sum(len(item['prediction']) for item in api_zs) / len(api_zs)
    api_cot_len = sum(len(item['prediction']) for item in api_cot) / len(api_cot)
    lora_zs_len = sum(len(item['prediction']) for item in lora_zs) / len(lora_zs)
    lora_cot_len = sum(len(item['prediction']) for item in lora_cot) / len(lora_cot)
    
    print(f"\n{'组别':<20} {'平均长度':>10} {'与参考答案比':>15}")
    print("-" * 50)
    print(f"{'参考答案':<20} {ref_len:>10.0f}字 {'-':>15}")
    print(f"{'API零样本':<20} {api_zs_len:>10.0f}字 {api_zs_len/ref_len:>14.1f}x")
    print(f"{'API-CoT':<20} {api_cot_len:>10.0f}字 {api_cot_len/ref_len:>14.1f}x")
    print(f"{'LoRA零样本':<20} {lora_zs_len:>10.0f}字 {lora_zs_len/ref_len:>14.1f}x")
    print(f"{'LoRA-CoT':<20} {lora_cot_len:>10.0f}字 {lora_cot_len/ref_len:>14.1f}x")
    print("-" * 50)
    
    print("\n💡 观察:")
    if api_cot_len > api_zs_len * 2:
        print(f"  - CoT生成的回答明显更长（{api_cot_len/api_zs_len:.1f}倍）")
        print(f"  - 这可能导致F1/ROUGE分数下降（关键词被稀释）")


if __name__ == "__main__":
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/comparison_100"
    
    # 先分析长度
    length_analysis(result_dir)
    
    print("\n" + "="*80)
    input("按Enter开始查看详细案例...")
    
    # 再看具体案例
    analyze_cases(result_dir, num_cases=5)
