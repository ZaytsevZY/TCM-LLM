#!/usr/bin/env python3
"""查看CoT答案提取效果"""
import json
import sys

result_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/comparison_v2/api_baseline/cot/predictions.json"

with open(result_file, 'r') as f:
    results = json.load(f)

print("=" * 80)
print("🔍 CoT答案提取效果检查")
print("=" * 80)

has_tags_count = sum(1 for r in results if r.get('has_answer_tags', False))
print(f"\n总样本数: {len(results)}")
print(f"使用答案标签: {has_tags_count} ({has_tags_count/len(results)*100:.1f}%)")
print(f"未使用标签: {len(results)-has_tags_count} ({(len(results)-has_tags_count)/len(results)*100:.1f}%)")

print("\n" + "=" * 80)
print("查看前3个案例:")

for i in range(min(3, len(results))):
    item = results[i]
    print(f"\n{'='*80}")
    print(f"案例 {i+1}")
    print(f"{'='*80}")
    
    print(f"\n【问题】")
    print(f"{item['full_question'][:150]}...")
    
    print(f"\n【参考答案】")
    print(f"{item['reference'][:150]}...")
    
    if 'raw_prediction' in item:
        print(f"\n【完整CoT输出】({len(item['raw_prediction'])}字)")
        print(f"{item['raw_prediction'][:300]}...")
        
        print(f"\n【提取的答案】({len(item['prediction'])}字)")
        print(f"{item['prediction'][:200]}...")
        
        print(f"\n【是否有标签】: {'✓ 是' if item.get('has_answer_tags') else '✗ 否'}")
    
    input("\n按Enter查看下一个...")

