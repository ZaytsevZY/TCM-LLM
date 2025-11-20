#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主评测脚本（支持并发）
支持零样本和CoT两种模式的评测
"""
import os
import sys
import argparse
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

# 本地模型配置
LOCAL_CONFIG = {
    "model_path": "Qwen/Qwen2.5-7B-Instruct",
    "lora_path": "./models/checkpoints/qwen2.5-7b-tcm-lora"
}


def main():
    parser = argparse.ArgumentParser(description='中医模型评测（支持并发）')
    parser.add_argument('--mode', type=str, default='api', 
                        choices=['local', 'api'],
                        help='评测模式: local(本地LoRA) 或 api(使用API)')
    parser.add_argument('--eval_file', type=str, default='data/evaluation/eval_100.json',
                        help='评测数据文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                        help='输出目录')
    parser.add_argument('--parallel', type=int, default=1,
                        help='并发数（仅API模式有效，推荐10）')
    parser.add_argument('--skip_zero_shot', action='store_true',
                        help='跳过零样本评测')
    parser.add_argument('--skip_cot', action='store_true',
                        help='跳过CoT评测')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 中医模型评测系统（并发版）")
    print("=" * 60)
    print(f"评测模式: {args.mode}")
    print(f"评测文件: {args.eval_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"并发数: {args.parallel}")
    print("")
    
    # 创建输出目录
    os.makedirs(f"{args.output_dir}/zero_shot", exist_ok=True)
    os.makedirs(f"{args.output_dir}/cot", exist_ok=True)
    
    # 加载评测数据
    print("📂 加载评测数据...")
    eval_data = load_eval_data(args.eval_file)
    print(f"✓ 已加载 {len(eval_data)} 条评测数据\n")
    
    # 初始化评测器
    if args.mode == "local":
        evaluator = ModelEvaluator(
            mode="local",
            model_path=LOCAL_CONFIG["model_path"],
            lora_path=LOCAL_CONFIG["lora_path"]
        )
        # 本地模式强制单线程
        num_workers = 1
        print("⚠️  本地模式仅支持单线程评测")
    else:
        evaluator = ModelEvaluator(
            mode="api",
            api_config=API_CONFIG
        )
        num_workers = args.parallel
    
    # ========================================
    # 零样本评测
    # ========================================
    if not args.skip_zero_shot:
        print("\n" + "=" * 60)
        print("1️⃣ 零样本评测")
        print("=" * 60)
        
        zero_shot_results = evaluator.batch_evaluate(
            eval_data=eval_data,
            prompt_builder=build_zero_shot_prompt,
            mode_name="零样本",
            max_tokens=2048,
            num_workers=num_workers
        )
        
        # 保存结果
        save_results(
            zero_shot_results,
            f"{args.output_dir}/zero_shot/predictions.json"
        )
        
        # 计算指标
        zero_shot_metrics = calculate_all_metrics(zero_shot_results)
        save_metrics(
            zero_shot_metrics,
            f"{args.output_dir}/zero_shot/metrics.json"
        )
        print_metrics(zero_shot_metrics, "零样本评测结果")
    
    # ========================================
    # CoT评测
    # ========================================
    if not args.skip_cot:
        print("\n" + "=" * 60)
        print("2️⃣ CoT评测")
        print("=" * 60)
        
        cot_results = evaluator.batch_evaluate(
            eval_data=eval_data,
            prompt_builder=build_cot_prompt,
            mode_name="CoT",
            max_tokens=4096,
            num_workers=num_workers
        )
        
        # 保存结果
        save_results(
            cot_results,
            f"{args.output_dir}/cot/predictions.json"
        )
        
        # 计算指标
        cot_metrics = calculate_all_metrics(cot_results)
        save_metrics(
            cot_metrics,
            f"{args.output_dir}/cot/metrics.json"
        )
        print_metrics(cot_metrics, "CoT评测结果")
    
    # ========================================
    # 对比分析
    # ========================================
    if not args.skip_zero_shot and not args.skip_cot:
        print("\n" + "=" * 60)
        print("📊 对比分析")
        print("=" * 60)
        
        print(f"\n{'指标':<20} {'零样本':>15} {'CoT':>15} {'提升':>15}")
        print("-" * 70)
        
        em_diff = cot_metrics['exact_match'] - zero_shot_metrics['exact_match']
        print(f"{'精确匹配率':<20} {zero_shot_metrics['exact_match']:>14.2%} "
              f"{cot_metrics['exact_match']:>14.2%} {em_diff:>+14.2%}")
        
        f1_diff = cot_metrics['avg_f1'] - zero_shot_metrics['avg_f1']
        print(f"{'平均F1':<20} {zero_shot_metrics['avg_f1']:>14.4f} "
              f"{cot_metrics['avg_f1']:>14.4f} {f1_diff:>+14.4f}")
        
        rouge_l_diff = (cot_metrics['rouge_scores']['rouge-l'] - 
                        zero_shot_metrics['rouge_scores']['rouge-l'])
        print(f"{'ROUGE-L':<20} {zero_shot_metrics['rouge_scores']['rouge-l']:>14.4f} "
              f"{cot_metrics['rouge_scores']['rouge-l']:>14.4f} {rouge_l_diff:>+14.4f}")
        
        time_diff = cot_metrics['avg_inference_time'] - zero_shot_metrics['avg_inference_time']
        print(f"{'平均推理时间(秒)':<20} {zero_shot_metrics['avg_inference_time']:>14.2f} "
              f"{cot_metrics['avg_inference_time']:>14.2f} {time_diff:>+14.2f}")
        
        print("-" * 70)
    
    print("\n" + "=" * 60)
    print("✅ 评测完成！")
    print("=" * 60)
    print(f"\n结果保存在: {args.output_dir}/")
    print("\n下一步:")
    print("  1. 查看详细结果: cat outputs/predictions/zero_shot/metrics.json")
    print("  2. 运行完整评测: python scripts/05_evaluate.py --eval_file data/evaluation/eval_500.json --parallel 10")
    print("  3. 生成分析报告: python scripts/06_analyze_results.py")


if __name__ == "__main__":
    main()
