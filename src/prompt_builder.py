#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt构建器（通用版）
"""

def format_question(instruction, input_text=""):
    """格式化完整问题"""
    instruction = instruction.strip()
    input_text = input_text.strip() if input_text else ""
    
    if input_text:
        return f"{instruction}\n\n补充信息：\n{input_text}"
    else:
        return instruction


def build_zero_shot_prompt(full_question):
    """构建零样本prompt（通用版）"""
    prompt = f"""你是一位专业的中医知识助手。请根据以下问题给出准确、专业的回答。

{full_question}

请给出你的回答："""
    
    return prompt


def build_cot_prompt(full_question):
    """构建CoT prompt（简化版）"""
    prompt = f"""你是一位专业的中医知识助手。请仔细分析以下问题，并给出详细的回答。

{full_question}

请按照以下思路分析和回答：
1. 理解问题的核心要求
2. 分析相关的中医理论或知识点
3. 给出清晰、准确的答案

请开始分析："""
    
    return prompt


# 测试代码
if __name__ == "__main__":
    print("测试Prompt构建器\n" + "=" * 60)
    
    # 测试1: 中医诊断
    print("\n测试1 - 中医诊断:")
    q1 = "患者咳嗽气喘3个月，舌苔白腻，脉滑数，请给出诊断和治疗建议。"
    print(build_zero_shot_prompt(q1))
    
    print("\n" + "-" * 60)
    
    # 测试2: 古文翻译
    print("\n测试2 - 古文翻译:")
    q2 = "将输入的古文翻译成现代文。\n\n补充信息：\n古文：是谓得时而调之。"
    print(build_zero_shot_prompt(q2))
