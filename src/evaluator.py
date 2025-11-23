#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测工具类（支持并发 + CoT答案提取）
"""
import json
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import openai
from tqdm import tqdm

class ModelEvaluator:
    """模型评测器"""
    
    def __init__(self, mode="local", model_path=None, lora_path=None, api_config=None):
        """初始化评测器"""
        self.mode = mode
        
        if mode == "local":
            print("🔧 加载本地LoRA模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 加载基座模型
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA权重
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                torch_dtype=torch.bfloat16
            )
            self.model.eval()
            print("✓ 模型加载完成")
            
        elif mode == "api":
            print("🔧 配置API评测...")
            self.client = openai.OpenAI(
                api_key=api_config["api_key"],
                base_url=api_config["base_url"]
            )
            self.api_model = api_config["model_name"]
            self.api_config = api_config
            print("✓ API配置完成")
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        """生成回答"""
        if self.mode == "local":
            return self._generate_local(prompt, max_tokens, temperature)
        elif self.mode == "api":
            return self._generate_api(prompt, max_tokens, temperature)
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """本地模型生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        return generated_text.strip()
    
    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """API生成（带重试）"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"API调用失败: {e}")
                    return ""
    
    def _evaluate_single(
        self,
        item: Dict[str, Any],
        prompt_builder,
        max_tokens: int,
        is_cot: bool = False
    ) -> Dict[str, Any]:
        """
        评测单个样本
        
        Args:
            item: 单个评测样本
            prompt_builder: prompt构建函数
            max_tokens: 最大生成token数
            is_cot: 是否是CoT模式（需要提取答案）
            
        Returns:
            评测结果
        """
        start_time = time.time()
        
        # 构建prompt
        prompt = prompt_builder(item["full_question"])
        
        # 生成回答
        raw_prediction = self.generate(prompt, max_tokens=max_tokens)
        
        inference_time = time.time() - start_time
        
        # 如果是CoT，提取答案标签
        if is_cot:
            from src.prompt_builder import extract_answer_from_cot
            extracted_answer, has_tags = extract_answer_from_cot(raw_prediction)
            
            result = {
                "id": item["id"],
                "instruction": item["instruction"],
                "input": item["input"],
                "full_question": item["full_question"],
                "reference": item["output"],
                "raw_prediction": raw_prediction,  # 保存完整输出
                "prediction": extracted_answer,     # 用于评测的答案
                "has_answer_tags": has_tags,        # 是否包含标签
                "inference_time": inference_time
            }
        else:
            # 零样本模式，直接使用原始输出
            result = {
                "id": item["id"],
                "instruction": item["instruction"],
                "input": item["input"],
                "full_question": item["full_question"],
                "reference": item["output"],
                "prediction": raw_prediction,
                "inference_time": inference_time
            }
        
        return result
    
    def batch_evaluate(
        self,
        eval_data: List[Dict[str, Any]],
        prompt_builder,
        mode_name: str,
        max_tokens: int = 2048,
        num_workers: int = 1,
        is_cot: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量评测（支持并发）
        
        Args:
            eval_data: 评测数据列表
            prompt_builder: prompt构建函数
            mode_name: 评测模式名称
            max_tokens: 最大生成token数
            num_workers: 并发线程数
            is_cot: 是否是CoT模式
            
        Returns:
            评测结果列表
        """
        results = []
        
        print(f"\n🔄 开始{mode_name}评测 ({len(eval_data)}条)...")
        print(f"并发数: {num_workers}")
        if is_cot:
            print("⚠️  CoT模式：将提取<答案>标签中的内容进行评测")
        
        if self.mode == "api" and num_workers > 1:
            results = self._batch_evaluate_parallel(
                eval_data, prompt_builder, mode_name, max_tokens, num_workers, is_cot
            )
        else:
            results = self._batch_evaluate_sequential(
                eval_data, prompt_builder, mode_name, max_tokens, is_cot
            )
        
        # 统计CoT标签使用情况
        if is_cot:
            has_tags_count = sum(1 for r in results if r.get('has_answer_tags', False))
            print(f"✓ {mode_name}评测完成 - {has_tags_count}/{len(results)} 条使用了答案标签")
        else:
            print(f"✓ {mode_name}评测完成")
        
        return results
    
    def _batch_evaluate_sequential(
        self,
        eval_data: List[Dict[str, Any]],
        prompt_builder,
        mode_name: str,
        max_tokens: int,
        is_cot: bool
    ) -> List[Dict[str, Any]]:
        """串行评测"""
        results = []
        for item in tqdm(eval_data, desc=f"{mode_name}评测"):
            result = self._evaluate_single(item, prompt_builder, max_tokens, is_cot)
            results.append(result)
        return results
    
    def _batch_evaluate_parallel(
        self,
        eval_data: List[Dict[str, Any]],
        prompt_builder,
        mode_name: str,
        max_tokens: int,
        num_workers: int,
        is_cot: bool
    ) -> List[Dict[str, Any]]:
        """并行评测（API模式）"""
        results = [None] * len(eval_data)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._evaluate_single, 
                    item, 
                    prompt_builder, 
                    max_tokens,
                    is_cot
                ): idx
                for idx, item in enumerate(eval_data)
            }
            
            with tqdm(total=len(eval_data), desc=f"{mode_name}评测(并发)") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"\n样本 {idx} 评测失败: {e}")
                        results[idx] = {
                            "id": eval_data[idx]["id"],
                            "prediction": "",
                            "inference_time": 0,
                            "error": str(e)
                        }
                    pbar.update(1)
        
        return results


def load_eval_data(file_path: str) -> List[Dict[str, Any]]:
    """加载评测数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存评测结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存: {output_path}")
