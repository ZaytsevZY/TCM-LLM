#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt构建器（标签化版本）
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
    """构建零样本prompt"""
    prompt = f"""你是一位专业的中医知识助手。请根据以下问题给出准确、专业的回答。

{full_question}

请给出你的回答："""
    
    return prompt


def build_cot_prompt(full_question):
    """构建CoT prompt（结构化输出）"""
    prompt = f"""你是一位专业的中医知识助手。请仔细分析以下问题，并给出详细的回答。

{full_question}

请按照以下格式输出（务必包含两个标签）：

<思考过程>
在这里写出你的分析思路：
1. 理解问题的核心要求
2. 分析相关的中医理论或知识点
3. 推导出答案
</思考过程>

<答案>
在这里给出简洁、准确的最终答案（不要重复思考过程）
</答案>

请开始："""
    
    return prompt


def extract_answer_from_cot(cot_output):
    """
    从CoT输出中提取答案标签内的内容
    
    Args:
        cot_output: CoT模型的完整输出
        
    Returns:
        extracted_answer: 提取的答案
        has_tags: 是否包含标签
    """
    import re
    
    # 尝试提取<答案>标签中的内容
    answer_pattern = r'<答案>(.*?)</答案>'
    match = re.search(answer_pattern, cot_output, re.DOTALL)
    
    if match:
        # 找到标签，提取答案
        extracted = match.group(1).strip()
        return extracted, True
    else:
        # 没有标签，返回原始输出
        return cot_output.strip(), False


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试Prompt构建器和答案提取")
    print("=" * 60)
    
    # 测试CoT prompt
    q1 = "患者咳嗽气喘3个月，舌苔白腻，脉滑数，请给出诊断。"
    print("\n【CoT Prompt】")
    print(build_cot_prompt(q1))
    
    # 测试答案提取
    print("\n" + "=" * 60)
    print("测试答案提取")
    print("=" * 60)
    
    # 测试用例1：有标签
    test_output_1 = """<思考过程>
根据患者症状，咳嗽气喘提示肺系疾病，舌苔白腻说明有痰湿，脉滑数提示痰热。
综合分析，考虑痰热壅肺证。
</思考过程>

<答案>
痰热壅肺证。治疗宜清热化痰，宣肺平喘。方用清金化痰汤加减。
</答案>"""
    
    extracted_1, has_tags_1 = extract_answer_from_cot(test_output_1)
    print(f"\n测试1 - 有标签")
    print(f"是否有标签: {has_tags_1}")
    print(f"提取的答案:\n{extracted_1}")
    
    # 测试用例2：无标签
    test_output_2 = "痰热壅肺证。治疗宜清热化痰，宣肺平喘。"
    
    extracted_2, has_tags_2 = extract_answer_from_cot(test_output_2)
    print(f"\n测试2 - 无标签")
    print(f"是否有标签: {has_tags_2}")
    print(f"提取的答案:\n{extracted_2}")
