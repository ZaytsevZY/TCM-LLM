#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测指标计算
包括准确率、F1、ROUGE等
"""
import json
from typing import List, Dict, Any
import jieba
from collections import Counter
from rouge_score import rouge_scorer

def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """计算精确匹配率"""
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions) if predictions else 0.0


def calculate_token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """
    计算token级别的F1分数
    
    Returns:
        {"precision": p, "recall": r, "f1": f1}
    """
    pred_tokens = set(jieba.lcut(prediction))
    ref_tokens = set(jieba.lcut(reference))
    
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    common = pred_tokens & ref_tokens
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    计算ROUGE分数
    
    Returns:
        {"rouge-1": r1, "rouge-2": r2, "rouge-l": rl}
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "rouge-1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        "rouge-2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
        "rouge-l": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    }


def calculate_avg_f1(results: List[Dict[str, Any]]) -> float:
    """计算平均F1分数"""
    f1_scores = []
    for result in results:
        f1_data = calculate_token_f1(result["prediction"], result["reference"])
        f1_scores.append(f1_data["f1"])
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def calculate_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算所有评测指标
    
    Args:
        results: 评测结果列表
        
    Returns:
        指标字典
    """
    predictions = [r["prediction"] for r in results]
    references = [r["reference"] for r in results]
    
    # 精确匹配
    exact_match = calculate_exact_match(predictions, references)
    
    # 平均F1
    avg_f1 = calculate_avg_f1(results)
    
    # ROUGE分数
    rouge_scores = calculate_rouge(predictions, references)
    
    # 平均推理时间
    avg_time = sum(r["inference_time"] for r in results) / len(results)
    
    metrics = {
        "exact_match": exact_match,
        "avg_f1": avg_f1,
        "rouge_scores": rouge_scores,
        "avg_inference_time": avg_time,
        "total_samples": len(results)
    }
    
    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """保存指标到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✓ 指标已保存: {output_path}")


def print_metrics(metrics: Dict[str, Any], title: str = "评测指标"):
    """打印指标"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"精确匹配率: {metrics['exact_match']:.2%}")
    print(f"平均F1分数: {metrics['avg_f1']:.4f}")
    print(f"ROUGE-1: {metrics['rouge_scores']['rouge-1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge_scores']['rouge-2']:.4f}")
    print(f"ROUGE-L: {metrics['rouge_scores']['rouge-l']:.4f}")
    print(f"平均推理时间: {metrics['avg_inference_time']:.2f}秒")
    print(f"{'='*60}\n")
