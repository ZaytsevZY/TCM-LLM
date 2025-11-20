#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测工具类（支持并发）
支持本地LoRA模型和API两种评测方式
"""
import json
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import openai
from tqdm import tqdm

class ModelEvaluator:
    """模型评测器"""
    
    def __init__(self, mode="local", model_path=None, lora_path=None, api_config=None):
        """
        初始化评测器
        
        Args:
            mode: "local" (本地LoRA模型) 或 "api" (API评测)
            model_path: 基座模型路径
            lora_path: LoRA模型路径
            api_config: API配置字典
        """
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
        """
        生成回答
        
        Args:
            prompt: 输入prompt
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
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
        
        # 只返回生成的部分
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
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    print(f"API调用失败: {e}")
                    return ""
    
    def _evaluate_single(
        self,
        item: Dict[str, Any],
        prompt_builder,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        评测单个样本
        
        Args:
            item: 单个评测样本
            prompt_builder: prompt构建函数
            max_tokens: 最大生成token数
            
        Returns:
            评测结果
        """
        start_time = time.time()
        
        # 构建prompt
        prompt = prompt_builder(item["full_question"])
        
        # 生成回答
        prediction = self.generate(prompt, max_tokens=max_tokens)
        
        inference_time = time.time() - start_time
        
        # 返回结果
        result = {
            "id": item["id"],
            "instruction": item["instruction"],
            "input": item["input"],
            "full_question": item["full_question"],
            "reference": item["output"],
            "prediction": prediction,
            "inference_time": inference_time
        }
        
        return result
    
    def batch_evaluate(
        self,
        eval_data: List[Dict[str, Any]],
        prompt_builder,
        mode_name: str,
        max_tokens: int = 2048,
        num_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """
        批量评测（支持并发）
        
        Args:
            eval_data: 评测数据列表
            prompt_builder: prompt构建函数
            mode_name: 评测模式名称（"zero_shot"或"cot"）
            max_tokens: 最大生成token数
            num_workers: 并发线程数（仅API模式有效）
            
        Returns:
            评测结果列表
        """
        results = []
        
        print(f"\n🔄 开始{mode_name}评测 ({len(eval_data)}条)...")
        print(f"并发数: {num_workers}")
        
        if self.mode == "api" and num_workers > 1:
            # API模式使用并发
            results = self._batch_evaluate_parallel(
                eval_data, prompt_builder, mode_name, max_tokens, num_workers
            )
        else:
            # 本地模式或单线程
            results = self._batch_evaluate_sequential(
                eval_data, prompt_builder, mode_name, max_tokens
            )
        
        print(f"✓ {mode_name}评测完成")
        return results
    
    def _batch_evaluate_sequential(
        self,
        eval_data: List[Dict[str, Any]],
        prompt_builder,
        mode_name: str,
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """串行评测"""
        results = []
        for item in tqdm(eval_data, desc=f"{mode_name}评测"):
            result = self._evaluate_single(item, prompt_builder, max_tokens)
            results.append(result)
        return results
    
    def _batch_evaluate_parallel(
        self,
        eval_data: List[Dict[str, Any]],
        prompt_builder,
        mode_name: str,
        max_tokens: int,
        num_workers: int
    ) -> List[Dict[str, Any]]:
        """并行评测（API模式）"""
        results = [None] * len(eval_data)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    self._evaluate_single, 
                    item, 
                    prompt_builder, 
                    max_tokens
                ): idx
                for idx, item in enumerate(eval_data)
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(eval_data), desc=f"{mode_name}评测(并发)") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"\n样本 {idx} 评测失败: {e}")
                        # 创建失败结果
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
