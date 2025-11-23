#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整对比实验（CoT答案提取版）
"""
import os
import sys
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import ModelEvaluator, load_eval_data, save_results
from src.metrics import calculate_all_metrics, save_metrics, print_metrics
from src.prompt_builder import build_zero_shot_prompt, build_cot_prompt

# API配置
API_CONFIG = {
    "api_key": "sk-aa792b68be91407f8ae2caf796988b7d",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model_name": "qwen-plus",
    "max_tokens": 4096,
    "temperature": 0.1
}

# 本地LoRA配置
LOCAL_CONFIG = {
    "model_path": "/home/zhayi/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
    "lora_path": "./models/checkpoints/qwen2.5-7b-tcm-lora"
}


def run_experiment(
    evaluator,
    eval_data,
    experiment_name,
    output_base_dir,
    num_workers=10
):
    """运行单个实验（零样本+CoT）"""
    print("\n" + "=" * 70)
    print(f"🧪 实验: {experiment_name}")
    print("=" * 70)
    
    exp_dir = f"{output_base_dir}/{experiment_name}"
    os.makedirs(f"{exp_dir}/zero_shot", exist_ok=True)
    os.makedirs(f"{exp_dir}/cot", exist_ok=True)
    
    results = {}
    
    # 1. 零样本评测
    print(f"\n[{experiment_name}] 1/2 零样本评测...")
    zero_shot_results = evaluator.batch_evaluate(
        eval_data=eval_data,
        prompt_builder=build_zero_shot_prompt,
        mode_name=f"{experiment_name}-零样本",
        max_tokens=2048,
        num_workers=num_workers,
        is_cot=False  # 零样本模式
    )
    
    save_results(zero_shot_results, f"{exp_dir}/zero_shot/predictions.json")
    zero_shot_metrics = calculate_all_metrics(zero_shot_results)
    save_metrics(zero_shot_metrics, f"{exp_dir}/zero_shot/metrics.json")
    print_metrics(zero_shot_metrics, f"{experiment_name} - 零样本")
    
    results['zero_shot'] = {
        'predictions': zero_shot_results,
        'metrics': zero_shot_metrics
    }
    
    # 2. CoT评测（提取答案标签）
    print(f"\n[{experiment_name}] 2/2 CoT评测（答案提取模式）...")
    cot_results = evaluator.batch_evaluate(
        eval_data=eval_data,
        prompt_builder=build_cot_prompt,
        mode_name=f"{experiment_name}-CoT",
        max_tokens=4096,
        num_workers=num_workers,
        is_cot=True  # ✨ CoT模式，会提取<答案>标签
    )
    
    save_results(cot_results, f"{exp_dir}/cot/predictions.json")
    cot_metrics = calculate_all_metrics(cot_results)
    save_metrics(cot_metrics, f"{exp_dir}/cot/metrics.json")
    print_metrics(cot_metrics, f"{experiment_name} - CoT")
    
    results['cot'] = {
        'predictions': cot_results,
        'metrics': cot_metrics
    }
    
    return results


def print_final_comparison(all_results):
    """打印最终对比"""
    print("\n" + "=" * 80)
    print("📊 最终对比分析")
    print("=" * 80)
    
    api_zs = all_results['api_baseline']['zero_shot']['metrics']
    api_cot = all_results['api_baseline']['cot']['metrics']
    lora_zs = all_results['lora_finetuned']['zero_shot']['metrics']
    lora_cot = all_results['lora_finetuned']['cot']['metrics']
    
    print(f"\n{'实验组':<25} {'精确匹配率':>12} {'平均F1':>12} {'ROUGE-L':>12} {'推理时间':>12}")
    print("-" * 80)
    
    print(f"{'API基线-零样本':<25} {api_zs['exact_match']:>11.2%} "
          f"{api_zs['avg_f1']:>11.4f} {api_zs['rouge_scores']['rouge-l']:>11.4f} "
          f"{api_zs['avg_inference_time']:>11.2f}s")
    
    print(f"{'API基线-CoT(提取)':<25} {api_cot['exact_match']:>11.2%} "
          f"{api_cot['avg_f1']:>11.4f} {api_cot['rouge_scores']['rouge-l']:>11.4f} "
          f"{api_cot['avg_inference_time']:>11.2f}s")
    
    print(f"{'LoRA微调-零样本':<25} {lora_zs['exact_match']:>11.2%} "
          f"{lora_zs['avg_f1']:>11.4f} {lora_zs['rouge_scores']['rouge-l']:>11.4f} "
          f"{lora_zs['avg_inference_time']:>11.2f}s")
    
    print(f"{'LoRA微调-CoT(提取)':<25} {lora_cot['exact_match']:>11.2%} "
          f"{lora_cot['avg_f1']:>11.4f} {lora_cot['rouge_scores']['rouge-l']:>11.4f} "
          f"{lora_cot['avg_inference_time']:>11.2f}s")
    
    print("-" * 80)
    
    # 关键对比
    print("\n关键发现:")
    
    # 1. 微调效果
    ft_improvement = lora_zs['avg_f1'] - api_zs['avg_f1']
    print(f"1. 微调效果: F1提升 {ft_improvement:+.4f} ({ft_improvement/api_zs['avg_f1']*100:+.1f}%)")
    
    # 2. CoT效果（API）
    cot_improvement_api = api_cot['avg_f1'] - api_zs['avg_f1']
    print(f"2. CoT效果-API: F1变化 {cot_improvement_api:+.4f} ({cot_improvement_api/api_zs['avg_f1']*100:+.1f}%)")
    
    # 3. CoT效果（LoRA）
    cot_improvement_lora = lora_cot['avg_f1'] - lora_zs['avg_f1']
    print(f"3. CoT效果-LoRA: F1变化 {cot_improvement_lora:+.4f} ({cot_improvement_lora/lora_zs['avg_f1']*100:+.1f}%)")
    
    # 4. 最佳组合
    best_f1 = max(api_zs['avg_f1'], api_cot['avg_f1'], 
                  lora_zs['avg_f1'], lora_cot['avg_f1'])
    best_group = ""
    if best_f1 == lora_cot['avg_f1']:
        best_group = "LoRA微调+CoT"
    elif best_f1 == lora_zs['avg_f1']:
        best_group = "LoRA微调+零样本"
    elif best_f1 == api_cot['avg_f1']:
        best_group = "API基线+CoT"
    else:
        best_group = "API基线+零样本"
    
    print(f"4. 最佳组合: {best_group} (F1={best_f1:.4f})")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='完整对比实验（CoT答案提取版）')
    parser.add_argument('--eval_file', type=str, default='data/evaluation/eval_100.json')
    parser.add_argument('--output_dir', type=str, default='outputs/comparison_v2')
    parser.add_argument('--parallel', type=int, default=10)
    parser.add_argument('--skip_api', action='store_true')
    parser.add_argument('--skip_lora', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 完整对比实验系统 v2（CoT答案提取）")
    print("=" * 80)
    print(f"评测文件: {args.eval_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"并发数: {args.parallel}")
    print("\n✨ 新功能: CoT会要求模型输出<答案>标签，只提取标签内容进行评测")
    print("")
    
    # 加载评测数据
    print("📂 加载评测数据...")
    eval_data = load_eval_data(args.eval_file)
    print(f"✓ 已加载 {len(eval_data)} 条数据\n")
    
    all_results = {}
    
    # 实验1: API基线
    if not args.skip_api:
        api_evaluator = ModelEvaluator(mode="api", api_config=API_CONFIG)
        all_results['api_baseline'] = run_experiment(
            evaluator=api_evaluator,
            eval_data=eval_data,
            experiment_name="api_baseline",
            output_base_dir=args.output_dir,
            num_workers=args.parallel
        )
    
    # 实验2: LoRA微调
    if not args.skip_lora:
        lora_evaluator = ModelEvaluator(
            mode="local",
            model_path=LOCAL_CONFIG["model_path"],
            lora_path=LOCAL_CONFIG["lora_path"]
        )
        all_results['lora_finetuned'] = run_experiment(
            evaluator=lora_evaluator,
            eval_data=eval_data,
            experiment_name="lora_finetuned",
            output_base_dir=args.output_dir,
            num_workers=1
        )
    
    # 最终对比
    if not args.skip_api and not args.skip_lora:
        print_final_comparison(all_results)
    
    # 保存完整结果
    summary = {
        'eval_file': args.eval_file,
        'total_samples': len(eval_data),
        'experiments': {}
    }
    
    for exp_name, exp_results in all_results.items():
        summary['experiments'][exp_name] = {
            'zero_shot': exp_results['zero_shot']['metrics'],
            'cot': exp_results['cot']['metrics']
        }
    
    with open(f"{args.output_dir}/summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完整结果已保存: {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
