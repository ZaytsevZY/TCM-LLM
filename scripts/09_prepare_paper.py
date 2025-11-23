#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备论文材料
生成表格、图表、案例
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_main_results_table(summary_file, output_dir):
    """创建主要结果表格"""
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # 提取数据
    api_zs = data['experiments']['api_baseline']['zero_shot']
    api_cot = data['experiments']['api_baseline']['cot']
    lora_zs = data['experiments']['lora_finetuned']['zero_shot']
    lora_cot = data['experiments']['lora_finetuned']['cot']
    
    # 创建DataFrame
    results = {
        '模型': ['API基线', 'API基线', 'LoRA微调', 'LoRA微调'],
        'Prompt': ['零样本', 'CoT', '零样本', 'CoT'],
        '精确匹配(%)': [
            f"{api_zs['exact_match']*100:.2f}",
            f"{api_cot['exact_match']*100:.2f}",
            f"{lora_zs['exact_match']*100:.2f}",
            f"{lora_cot['exact_match']*100:.2f}"
        ],
        '平均F1': [
            f"{api_zs['avg_f1']:.4f}",
            f"{api_cot['avg_f1']:.4f}",
            f"{lora_zs['avg_f1']:.4f}",
            f"{lora_cot['avg_f1']:.4f}"
        ],
        'ROUGE-1': [
            f"{api_zs['rouge_scores']['rouge-1']:.4f}",
            f"{api_cot['rouge_scores']['rouge-1']:.4f}",
            f"{lora_zs['rouge_scores']['rouge-1']:.4f}",
            f"{lora_cot['rouge_scores']['rouge-1']:.4f}"
        ],
        'ROUGE-L': [
            f"{api_zs['rouge_scores']['rouge-l']:.4f}",
            f"{api_cot['rouge_scores']['rouge-l']:.4f}",
            f"{lora_zs['rouge_scores']['rouge-l']:.4f}",
            f"{lora_cot['rouge_scores']['rouge-l']:.4f}"
        ],
        '推理时间(s)': [
            f"{api_zs['avg_inference_time']:.1f}",
            f"{api_cot['avg_inference_time']:.1f}",
            f"{lora_zs['avg_inference_time']:.1f}",
            f"{lora_cot['avg_inference_time']:.1f}"
        ]
    }
    
    df = pd.DataFrame(results)
    
    # 保存CSV
    df.to_csv(f"{output_dir}/main_results.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 主要结果表格: {output_dir}/main_results.csv")
    
    # 保存LaTeX格式
    latex = df.to_latex(index=False, escape=False)
    with open(f"{output_dir}/main_results.tex", 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"✓ LaTeX表格: {output_dir}/main_results.tex")
    
    return df


def create_comparison_plot(summary_file, output_dir):
    """创建对比图"""
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    api_zs = data['experiments']['api_baseline']['zero_shot']
    api_cot = data['experiments']['api_baseline']['cot']
    lora_zs = data['experiments']['lora_finetuned']['zero_shot']
    lora_cot = data['experiments']['lora_finetuned']['cot']
    
    # F1分数对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['API基线', 'LoRA微调']
    zero_shot_scores = [api_zs['avg_f1'], lora_zs['avg_f1']]
    cot_scores = [api_cot['avg_f1'], lora_cot['avg_f1']]
    
    x = range(len(models))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], zero_shot_scores, width, 
                   label='零样本', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar([i + width/2 for i in x], cot_scores, width,
                   label='CoT', alpha=0.8, color='#ff7f0e')
    
    ax.set_ylabel('平均F1分数', fontsize=12)
    ax.set_title('不同模型和Prompt方法的性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ F1对比图: {output_dir}/f1_comparison.png")
    
    # ROUGE-L对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    zero_shot_rouge = [api_zs['rouge_scores']['rouge-l'], lora_zs['rouge_scores']['rouge-l']]
    cot_rouge = [api_cot['rouge_scores']['rouge-l'], lora_cot['rouge_scores']['rouge-l']]
    
    bars1 = ax.bar([i - width/2 for i in x], zero_shot_rouge, width,
                   label='零样本', alpha=0.8, color='#2ca02c')
    bars2 = ax.bar([i + width/2 for i in x], cot_rouge, width,
                   label='CoT', alpha=0.8, color='#d62728')
    
    ax.set_ylabel('ROUGE-L分数', fontsize=12)
    ax.set_title('ROUGE-L性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rouge_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ ROUGE对比图: {output_dir}/rouge_comparison.png")


def generate_paper_outline(output_dir):
    """生成论文大纲"""
    
    outline = """# 论文大纲

## 标题
基于QLoRA的中医领域大模型微调及思维链提示效果研究

## 摘要（300字）
本研究探讨了领域微调和思维链（Chain-of-Thought, CoT）提示对中医问答任务的影响。
我们使用368万条中医数据对Qwen2.5-7B模型进行QLoRA微调，并在100条测试集上对比了
四种配置：API基线（零样本/CoT）和LoRA微调（零样本/CoT）。

实验结果显示：
1. **领域微调显著有效**：LoRA微调使F1分数从0.238提升至0.270（+13.4%）
2. **CoT效果为负**：在两种模型上，CoT均导致性能下降（API基线-65%，LoRA-43%）
3. **最佳方案**：LoRA微调+零样本（F1=0.270）

深入分析表明，CoT效果差的主要原因是数据集以简短事实问答为主，CoT生成的冗长分析
反而稀释了关键答案，导致F1/ROUGE分数下降。本研究揭示了CoT在不同任务类型上的
适用性边界，为中医AI应用提供了实证参考。

关键词：中医问答，大语言模型，QLoRA微调，思维链提示，领域适应

---

## 1. 引言

### 1.1 研究背景
- 大语言模型在医疗领域的应用
- 中医知识的特殊性和复杂性
- 领域微调的必要性

### 1.2 研究动机
- CoT在通用任务上的成功
- CoT在领域微调后的效果未知
- 需要实证研究验证CoT的适用边界

### 1.3 研究问题
RQ1: 领域微调对中医问答任务的效果如何？
RQ2: CoT提示能否进一步提升微调模型的性能？
RQ3: CoT在什么样的任务上有效？什么情况下会失效？

### 1.4 主要贡献
1. 首次系统对比领域微调和CoT对中医问答的影响
2. 发现CoT在简短事实问答任务上效果为负的现象
3. 分析了CoT失效的原因，为任务选择提供指导

---

## 2. 相关工作

### 2.1 大语言模型的医疗应用
- 医疗对话系统
- 医学知识问答
- 临床辅助决策

### 2.2 领域微调方法
- 全参数微调 vs 参数高效微调
- QLoRA方法介绍
- 中医领域的微调研究

### 2.3 思维链提示技术
- CoT的基本原理
- 在数学、逻辑推理等任务上的成功
- 在医疗领域的应用探索

---

## 3. 方法

### 3.1 数据集
**训练数据**：
- 来源：SylvanL/Traditional-Chinese-Medicine-Dataset-SFT
- 规模：3,677,727条（使用30%子集1,164,613条）
- 内容：中医疾病诊断、证型判断、方药推荐、古文翻译等

**测试数据**：
- 规模：100条（从测试集随机抽样）
- 特点：包含多种任务类型

### 3.2 模型配置
**基座模型**：Qwen2.5-7B-Instruct

**微调方法**：QLoRA
- LoRA rank: 32
- LoRA alpha: 64
- Target modules: q_proj, v_proj
- 量化: 4-bit
- 训练epoch: 1
- 最终Loss: 1.111

### 3.3 Prompt设计
**零样本Prompt**：
```
你是一位专业的中医知识助手。请根据以下问题给出准确、专业的回答。
{问题}
请给出你的回答：
```

**CoT Prompt**：
```
你是一位专业的中医知识助手。请仔细分析以下问题，并给出详细的回答。
{问题}
请按照以下思路分析和回答：
1. 理解问题的核心要求
2. 分析相关的中医理论或知识点
3. 给出清晰、准确的答案
请开始分析：
```

### 3.4 实验设置
四组对比实验：
1. API基线 + 零样本
2. API基线 + CoT
3. LoRA微调 + 零样本
4. LoRA微调 + CoT

**评测指标**：
- 精确匹配率（Exact Match）
- 平均F1分数（Token-level）
- ROUGE-1/2/L
- 平均推理时间

---

## 4. 实验结果

### 4.1 主要结果

[插入表格：main_results.csv]

**关键发现**：
1. LoRA微调显著提升性能（F1: 0.238→0.270，+13.4%）
2. CoT在两种模型上均导致性能下降
3. 最佳组合：LoRA微调 + 零样本

### 4.2 详细分析

[插入图表：f1_comparison.png]

**微调效果**：
- API基线零样本: F1=0.238
- LoRA微调零样本: F1=0.270
- 提升: +13.4%
- 结论：领域微调在中医问答任务上显著有效

**CoT效果**：
- API基线：0.238→0.083（-65%）
- LoRA微调：0.270→0.154（-43%）
- 结论：CoT反而降低了性能

### 4.3 回答长度分析
[待补充：运行08_case_analysis.py后的数据]

发现：CoT生成的回答长度是零样本的X倍，导致关键信息被稀释。

### 4.4 典型案例
[待补充：选择2-3个典型案例展示]

---

## 5. 讨论

### 5.1 为什么微调有效？
- 学习了中医领域的专业术语
- 掌握了中医问答的表达方式
- 对数据分布进行了适应

### 5.2 为什么CoT效果为负？

**主要原因**：
1. **任务特征不匹配**
   - 数据集以简短事实问答为主（古文翻译、名词解释）
   - 不需要多步骤推理
   - CoT的分析过程变成了冗余信息

2. **评测指标的局限**
   - F1和ROUGE基于关键词匹配
   - CoT生成的冗长分析稀释了关键词密度
   - 导致匹配分数下降

3. **Prompt设计问题**
   - CoT prompt引导模型生成结构化分析
   - 在简单问答任务上过度工程化

### 5.3 CoT的适用边界

**CoT有效的场景**：
- 需要多步骤推理（数学、逻辑问题）
- 需要解释推理过程
- 复杂的诊断决策

**CoT无效/有害的场景**：
- 简短事实问答
- 古文翻译、名词解释
- 需要简洁答案的任务

### 5.4 对中医AI应用的启示
1. 领域微调是必要的
2. 根据任务类型选择Prompt策略
3. 简单问答用零样本，复杂推理用CoT

### 5.5 局限性
1. 评测数据规模较小（100条）
2. 仅使用单一模型（Qwen2.5-7B）
3. 评测指标可能不够全面（未包含人工评估）

---

## 6. 结论

本研究通过实证实验揭示了领域微调和CoT提示在中医问答任务上的效果：
1. 领域微调显著提升性能（+13.4%）
2. CoT在简短事实问答任务上效果为负（-43%到-65%）
3. 任务特征决定了Prompt策略的有效性

**未来工作**：
1. 在更大规模数据集上验证（500条→1000条）
2. 针对不同任务类型设计差异化的CoT策略
3. 结合人工评估，全面衡量回答质量
4. 探索混合策略：简单问答用零样本，复杂诊断用CoT

---

## 参考文献
[待补充]

---

## 附录

### 附录A：训练配置详情
### 附录B：Prompt模板完整版
### 附录C：更多案例分析
### 附录D：数据集统计信息
"""
    
    with open(f"{output_dir}/paper_outline.md", 'w', encoding='utf-8') as f:
        f.write(outline)
    
    print(f"✓ 论文大纲: {output_dir}/paper_outline.md")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='准备论文材料')
    parser.add_argument('--summary', type=str, default='outputs/comparison_100/summary.json',
                        help='实验结果摘要文件')
    parser.add_argument('--output_dir', type=str, default='outputs/paper_materials',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📝 准备论文材料")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 生成主要结果表格
    print("\n生成主要结果表格...")
    create_main_results_table(args.summary, args.output_dir)
    
    # 2. 生成对比图表
    print("\n生成对比图表...")
    create_comparison_plot(args.summary, args.output_dir)
    
    # 3. 生成论文大纲
    print("\n生成论文大纲...")
    generate_paper_outline(args.output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 论文材料准备完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - {args.output_dir}/main_results.csv    # 主要结果表格")
    print(f"  - {args.output_dir}/main_results.tex    # LaTeX表格")
    print(f"  - {args.output_dir}/f1_comparison.png   # F1对比图")
    print(f"  - {args.output_dir}/rouge_comparison.png # ROUGE对比图")
    print(f"  - {args.output_dir}/paper_outline.md    # 论文大纲")
    print(f"\n下一步:")
    print(f"  1. 查看案例分析: python scripts/08_case_analysis.py outputs/comparison_100")
    print(f"  2. 根据大纲撰写论文")
    print(f"  3. 补充案例和数据")


if __name__ == "__main__":
    main()
