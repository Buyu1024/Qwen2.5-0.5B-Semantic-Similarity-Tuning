"""
推理与评估脚本 — RTX 5090 优化版

1. 加载 LoRA 微调后的模型（Flash Attention 2 + torch.compile）
2. 单条/批量推理
3. 与 BERT-base-chinese + 分类头基线对比
4. 零样本基线评估
"""

import os
import sys
import json
import time
import re
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime
from functools import lru_cache

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    pipeline,
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from config import (
    MODEL_PATH, OUTPUT_DIR, DATA_DIR,
    SYSTEM_PROMPT, USER_TEMPLATE, LABEL_MAP,
)


class SemanticSimilarityInference:
    """语义相似度推理器（LoRA 微调后的 Qwen2.5）— 性能优化版"""

    def __init__(self, lora_path: str = OUTPUT_DIR, use_compile: bool = False):
        print(f"加载基础模型: {MODEL_PATH}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 检测 Flash Attention 2
        attn_impl = "sdpa"
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print(f"🚀 使用 Flash Attention 2 ({flash_attn.__version__})")
        except ImportError:
            print("📌 使用 SDPA (scaled_dot_product_attention)")

        # 加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation=attn_impl,
        )

        print(f"加载 LoRA 适配器: {lora_path}")
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        # torch.compile（推理加速）
        if use_compile and hasattr(torch, "compile"):
            print("🔥 torch.compile 推理优化...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"⚠️  编译失败: {e}")

        self.model.eval()
        self.model.config.use_cache = True  # 推理时恢复 KV cache
        print("模型加载完成！\n")

    def predict_single(
        self,
        sentence1: str,
        sentence2: str,
        max_new_tokens: int = 5,
    ) -> Tuple[str, int, float]:
        """
        预测单对句子的语义等价性

        Returns:
            (label_text, label_id, confidence)
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    sentence1=sentence1,
                    sentence2=sentence2,
                ),
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # 贪心解码，保证确定性
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # 解码生成的文本
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 解析结果
        label_id, confidence = self._parse_result(generated_text, outputs)

        return generated_text, label_id, confidence

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 8,
    ) -> List[dict]:
        """批量预测"""
        results = []
        total = len(pairs)

        for i in range(0, total, batch_size):
            batch = pairs[i:i+batch_size]
            for s1, s2 in batch:
                gen_text, label_id, conf = self.predict_single(s1, s2)
                results.append({
                    "sentence1": s1,
                    "sentence2": s2,
                    "generated": gen_text,
                    "prediction": label_id,
                    "confidence": conf,
                    "pred_label_text": LABEL_MAP[label_id] if label_id in LABEL_MAP else "未知",
                })

            if (i + batch_size) % 100 == 0:
                print(f"  已处理 {min(i + batch_size, total)}/{total}...")

        return results

    def _parse_result(self, generated_text: str, outputs) -> Tuple[int, float]:
        """
        解析生成的文本，提取标签和置信度

        优先匹配"等价"/"不等价"，支持多种输出格式
        """
        text = generated_text.strip()

        # 尝试直接匹配
        if "等价" in text and "不等价" not in text:
            label_id = 1
        elif "不等价" in text:
            label_id = 0
        elif "不同" in text or "不相似" in text or "不匹配" in text:
            label_id = 0
        elif "相同" in text or "相似" in text or "匹配" in text:
            label_id = 1
        else:
            # 无法解析，默认预测为 0（保守估计）
            label_id = 0

        # 计算置信度（从生成概率获取，如果有 scores）
        confidence = 1.0
        if outputs.scores and len(outputs.scores) > 0:
            # 取第一个生成 token 的 top-1 概率
            first_token_scores = outputs.scores[0][0]  # (vocab_size,)
            probs = torch.softmax(first_token_scores, dim=-1)
            confidence = float(probs.max().cpu())

        return label_id, confidence

    def interactive(self):
        """交互式推理模式"""
        print("=" * 50)
        print("语义相似度判断 - 交互模式")
        print("输入两个句子，判断是否语义等价")
        print("输入 'quit' 退出\n")
        print("=" * 50)

        while True:
            s1 = input("\n句子1: ").strip()
            if s1.lower() == "quit":
                break
            s2 = input("句子2: ").strip()
            if s2.lower() == "quit":
                break

            if not s1 or not s2:
                print("句子不能为空！")
                continue

            start = time.time()
            gen_text, label_id, conf = self.predict_single(s1, s2)
            elapsed = time.time() - start

            label_str = "✅ 等价" if label_id == 1 else "❌ 不等价"
            print(f"\n结果: {label_str}")
            print(f"原始输出: {gen_text}")
            print(f"置信度: {conf:.4f}")
            print(f"耗时: {elapsed:.3f}s")


class BertBaseline:
    """BERT-base-chinese + 分类头基线模型"""

    def __init__(self, bert_path: str = None):
        if bert_path is None:
            bert_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bert-base-chinese")

        from transformers import BertTokenizer, BertForSequenceClassification

        print(f"加载 BERT 基线模型: {bert_path}")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels=2,
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        print("BERT 基线模型加载完成！\n")

    def predict(self, sentence1: str, sentence2: str) -> Tuple[int, float]:
        """预测单对句子"""
        inputs = self.tokenizer(
            sentence1, sentence2,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).cpu())
            conf = float(probs.max().cpu())

        return pred, conf


def evaluate_on_test_set():
    """在测试集上评估微调后的模型"""
    print("=" * 60)
    print("测试集评估")
    print("=" * 60)

    # 加载测试数据
    from datasets import DatasetDict, load_from_disk

    data_path = os.path.join(DATA_DIR, "processed")
    if os.path.exists(data_path):
        dataset = DatasetDict.load_from_disk(data_path)
    else:
        from preprocess import prepare_dataset
        dataset = prepare_dataset()

    test_data = dataset["test"]

    # 提取原始 sentence pair 和 label
    # 注意：processed 数据集是 messages 格式
    test_pairs = []
    test_labels = []

    for item in test_data:
        messages = item["messages"]
        # user message 包含两个句子
        # 从 message 中提取原始 label
        assistant_content = messages[-1]["content"]
        label = 1 if "等价" in assistant_content else 0

        test_labels.append(label)
        # 不需要提取句子对，用 messages 直接推理

    print(f"测试集大小: {len(test_labels)}")
    print(f"正样本比例: {sum(test_labels)/len(test_labels):.2%}")

    # 加载模型
    infer = SemanticSimilarityInference()
    infer.model.config.use_cache = True  # 推理时恢复 cache

    # 批量推理
    results = []
    # 恢复 sentence pairs
    test_pairs = []
    for item in test_data:
        messages = item["messages"]
        user_msg = messages[1]["content"]
        # 从 user template 中提取两个句子
        # 格式: "句子1：xxx\n句子2：xxx\n\n请判断..."
        lines = user_msg.split("\n")
        s1 = lines[0].replace("句子1：", "").strip() if lines else ""
        s2 = lines[1].replace("句子2：", "").strip() if len(lines) > 1 else ""
        test_pairs.append((s1, s2))

    print("运行推理...")
    results = infer.predict_batch(test_pairs)

    # 计算指标
    pred_labels = [r["prediction"] for r in results]
    confidences = [r["confidence"] for r in results]

    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)

    acc = accuracy_score(test_labels, pred_labels)
    prec = precision_score(test_labels, pred_labels, zero_division=0)
    rec = recall_score(test_labels, pred_labels, zero_division=0)
    f1 = f1_score(test_labels, pred_labels, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # AUC（使用 confidence 作为概率）
    try:
        auc = roc_auc_score(test_labels, confidences)
        print(f"AUC-ROC:   {auc:.4f}")
    except Exception:
        print("AUC-ROC:   N/A")

    print("\n分类报告:")
    print(classification_report(
        test_labels, pred_labels,
        target_names=["不等价", "等价"],
        zero_division=0,
    ))

    # 保存结果
    result_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc if "auc" in dir() else None,
            "total_samples": len(test_labels),
            "positive_ratio": sum(test_labels) / len(test_labels),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {result_path}")

    # 打印一些错误案例
    print("\n--- 错误预测示例 (前10条) ---")
    errors = []
    for i, (true_l, pred_l) in enumerate(zip(test_labels, pred_labels)):
        if true_l != pred_l:
            errors.append((i, true_l, pred_l, test_pairs[i], results[i]["generated"]))

    for i, true_l, pred_l, (s1, s2), gen_text in errors[:10]:
        true_text = "等价" if true_l == 1 else "不等价"
        pred_text = "等价" if pred_l == 1 else "不等价"
        print(f"  [{i}] 真实:{true_text} 预测:{pred_text}")
        print(f"    句1: {s1[:50]}...")
        print(f"    句2: {s2[:50]}...")
        print(f"    生成: {gen_text[:50]}")
        print()


def zero_shot_baseline():
    """Qwen2.5-0.5B-Instruct 零样本基线（不做微调）"""
    print("=" * 60)
    print("零样本基线评估（加载原始模型，不加载 LoRA）")
    print("=" * 60)

    # 检测 Flash Attention 2
    attn_impl = "sdpa"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    # 加载原始模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation=attn_impl,
    )
    model.eval()

    # 加载少量测试数据
    from datasets import DatasetDict, load_from_disk
    data_path = os.path.join(DATA_DIR, "processed")
    if os.path.exists(data_path):
        dataset = DatasetDict.load_from_disk(data_path)
    else:
        from preprocess import prepare_dataset
        dataset = prepare_dataset()

    test_data = dataset["test"]

    # 提取
    test_pairs = []
    test_labels = []
    for item in test_data:
        messages = item["messages"]
        label = 1 if "等价" in messages[-1]["content"] else 0
        test_labels.append(label)

        user_msg = messages[1]["content"]
        lines = user_msg.split("\n")
        s1 = lines[0].replace("句子1：", "").strip() if lines else ""
        s2 = lines[1].replace("句子2：", "").strip() if len(lines) > 1 else ""
        test_pairs.append((s1, s2))

    # 只评估前1000条（省钱时间）
    subset_size = min(1000, len(test_pairs))
    test_pairs = test_pairs[:subset_size]
    test_labels = test_labels[:subset_size]
    print(f"评估样本: {subset_size}")

    pred_labels = []
    for i, (s1, s2) in enumerate(test_pairs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(sentence1=s1, sentence2=s2)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][len(inputs.input_ids[0]):]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if "等价" in gen_text and "不等价" not in gen_text:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i+1}/{subset_size}")

    acc = accuracy_score(test_labels, pred_labels)
    prec = precision_score(test_labels, pred_labels, zero_division=0)
    rec = recall_score(test_labels, pred_labels, zero_division=0)
    f1 = f1_score(test_labels, pred_labels, zero_division=0)

    print(f"\n零样本基线结果 ({subset_size} 样本):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="语义相似度推理")
    parser.add_argument(
        "--mode",
        choices=["interactive", "evaluate", "baseline"],
        default="interactive",
        help="运行模式",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=OUTPUT_DIR,
        help="LoRA 适配器路径",
    )
    parser.add_argument(
        "--s1", type=str, help="句子1（仅在非 interactive 模式下使用）"
    )
    parser.add_argument(
        "--s2", type=str, help="句子2（仅在非 interactive 模式下使用）"
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        infer = SemanticSimilarityInference(args.lora_path)
        infer.interactive()
    elif args.mode == "evaluate":
        evaluate_on_test_set()
    elif args.mode == "baseline":
        zero_shot_baseline()
