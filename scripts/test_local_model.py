#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地LoRA模型快速测试脚本
测试模型加载、推理、输出格式等基本功能
"""
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置
BASE_MODEL_PATH = "/home/zhayi/.cache/modelscope/hub/Qwen/Qwen2___5-7B-Instruct"
LORA_PATH = "./models/checkpoints/qwen2.5-7b-tcm-lora"

# 测试样例
TEST_CASES = [
    {
        "question": "患者主诉：咳嗽、咳痰1周，加重3天。现症：咳嗽频繁，痰色黄稠，难以咳出，伴有发热，体温38.5℃。舌质红，苔黄腻，脉滑数。请问应诊断为何证型？",
        "expected_pattern": "痰热壅肺",
        "category": "辨证"
    },
    {
        "question": "患者女性，35岁，月经不调，经期延后，量少色淡，伴有腰膝酸软，头晕耳鸣。舌淡苔白，脉沉细。请开具处方。",
        "expected_keywords": ["熟地黄", "当归", "白芍"],
        "category": "开方"
    },
    {
        "question": "麻黄的功效是什么？",
        "expected_keywords": ["发汗", "平喘", "利水"],
        "category": "药性"
    }
]


def print_separator(char="=", length=80):
    """打印分隔线"""
    print(char * length)


def print_section(title):
    """打印章节标题"""
    print_separator()
    print(f"🔍 {title}")
    print_separator()


def load_model():
    """加载模型和LoRA权重"""
    print_section("步骤 1/4: 加载模型")
    
    try:
        print(f"📂 基础模型路径: {BASE_MODEL_PATH}")
        print(f"📂 LoRA权重路径: {LORA_PATH}")
        print()
        
        # 检查路径
        if not os.path.exists(BASE_MODEL_PATH):
            print(f"❌ 基础模型路径不存在: {BASE_MODEL_PATH}")
            print("💡 请修改脚本中的 BASE_MODEL_PATH")
            sys.exit(1)
            
        if not os.path.exists(LORA_PATH):
            print(f"❌ LoRA权重路径不存在: {LORA_PATH}")
            print("💡 请先运行训练脚本生成LoRA权重")
            sys.exit(1)
        
        # 加载tokenizer
        print("⏳ 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )
        print("✓ Tokenizer 加载成功")
        
        # 加载基础模型
        print("⏳ 加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ 基础模型加载成功")
        
        # 加载LoRA权重
        print("⏳ 加载 LoRA 权重...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        model = model.merge_and_unload()  # 合并权重以提高推理速度
        print("✓ LoRA 权重加载成功")
        
        # 设置为评估模式
        model.eval()
        
        # 打印模型信息
        print()
        print("📊 模型信息:")
        print(f"  - 设备: {next(model.parameters()).device}")
        print(f"  - 数据类型: {next(model.parameters()).dtype}")
        print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_basic_generation(model, tokenizer):
    """测试基本生成能力"""
    print_section("步骤 2/4: 基本生成测试")
    
    test_prompt = "你好"
    
    print(f"输入: {test_prompt}")
    print()
    
    try:
        # 构建消息
        messages = [{"role": "user", "content": test_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成
        print("⏳ 生成中...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                top_p=0.9
            )
        
        inference_time = time.time() - start_time
        
        # 解码
        response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        print(f"输出: {response}")
        print(f"⏱️  推理时间: {inference_time:.2f}秒")
        print("✓ 基本生成测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本生成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tcm_cases(model, tokenizer):
    """测试中医案例"""
    print_section("步骤 3/4: 中医案例测试")
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'─' * 80}")
        print(f"📋 案例 {i}/{len(TEST_CASES)}: {case['category']}")
        print(f"{'─' * 80}")
        
        print(f"问题:\n{case['question']}\n")
        
        try:
            # 构建消息
            messages = [{"role": "user", "content": case['question']}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # 生成
            print("⏳ 生成中...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9
                )
            
            inference_time = time.time() - start_time
            
            # 解码
            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            
            print(f"回答:\n{response}\n")
            print(f"⏱️  推理时间: {inference_time:.2f}秒")
            
            # 简单验证
            check_passed = True
            
            if 'expected_pattern' in case:
                if case['expected_pattern'] in response:
                    print(f"✓ 包含预期证型: {case['expected_pattern']}")
                else:
                    print(f"⚠️  未找到预期证型: {case['expected_pattern']}")
                    check_passed = False
            
            if 'expected_keywords' in case:
                found_keywords = [kw for kw in case['expected_keywords'] if kw in response]
                if found_keywords:
                    print(f"✓ 包含关键词: {', '.join(found_keywords)}")
                else:
                    print(f"⚠️  未找到任何预期关键词: {', '.join(case['expected_keywords'])}")
                    check_passed = False
            
            if check_passed:
                print("✓ 案例测试通过")
                passed += 1
            else:
                print("⚠️  案例测试部分通过")
                failed += 1
                
        except Exception as e:
            print(f"❌ 案例测试失败: {str(e)}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n{'─' * 80}")
    print(f"📊 测试结果: {passed}/{len(TEST_CASES)} 通过")
    print(f"{'─' * 80}")
    
    return passed, failed


def test_performance(model, tokenizer):
    """性能测试"""
    print_section("步骤 4/4: 性能测试")
    
    test_prompt = "患者出现头痛、发热症状，请进行辨证分析。"
    num_runs = 5
    
    print(f"测试提示: {test_prompt}")
    print(f"运行次数: {num_runs}")
    print()
    
    times = []
    
    try:
        for i in range(num_runs):
            # 构建消息
            messages = [{"role": "user", "content": test_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True
                )
            
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            print(f"  第 {i+1} 次: {inference_time:.2f}秒")
        
        print()
        print(f"📊 性能统计:")
        print(f"  - 平均时间: {sum(times)/len(times):.2f}秒")
        print(f"  - 最快: {min(times):.2f}秒")
        print(f"  - 最慢: {max(times):.2f}秒")
        print("✓ 性能测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print_separator("=")
    print("🚀 本地LoRA模型测试系统")
    print_separator("=")
    print()
    
    # 步骤1: 加载模型
    model, tokenizer = load_model()
    print()
    
    # 步骤2: 基本生成测试
    if not test_basic_generation(model, tokenizer):
        print("\n❌ 基本生成测试失败，终止测试")
        sys.exit(1)
    print()
    
    # 步骤3: 中医案例测试
    passed, failed = test_tcm_cases(model, tokenizer)
    print()
    
    # 步骤4: 性能测试
    test_performance(model, tokenizer)
    print()
    
    # 最终总结
    print_separator("=")
    print("📋 测试总结")
    print_separator("=")
    print(f"✓ 模型加载: 成功")
    print(f"✓ 基本生成: 成功")
    print(f"✓ 中医案例: {passed}/{len(TEST_CASES)} 通过")
    print(f"✓ 性能测试: 完成")
    print()
    
    if failed == 0:
        print("🎉 所有测试通过！模型工作正常")
        print("\n下一步:")
        print("  python scripts/07_full_comparison.py --eval_file data/evaluation/eval_100.json")
    else:
        print(f"⚠️  {failed} 个案例未完全通过，建议检查模型输出质量")
        print("\n可能原因:")
        print("  1. 训练数据不足")
        print("  2. 训练轮数不够")
        print("  3. 学习率设置不当")
    
    print_separator("=")


if __name__ == "__main__":
    main()