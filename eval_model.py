"""
推理评估脚本 — 加载 LoRA 适配器，在验证集上测试准确率

用法: python3 eval_model.py
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-0.5B-Instruct")
LORA_PATH  = os.path.join(BASE_DIR, "outputs", "lora")
DEV_FILE   = os.path.join(BASE_DIR, "data", "BQ_Corpus", "dev.csv")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_model():
    """加载基座模型 + LoRA 适配器"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    log("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    log(f"加载 LoRA 适配器: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    model.config.use_cache = True  # 推理时启用 KV cache

    log(f"模型加载完成，设备: {model.device}")
    return model, tokenizer


def load_test_data(n_samples: int = 500):
    """加载验证集，随机采样"""
    df = pd.read_csv(DEV_FILE)
    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
    df = df.reset_index(drop=True)

    log(f"测试数据: {len(df)} 条 (正样本 {df['label'].mean():.1%})")
    return df


@torch.no_grad()
def predict(model, tokenizer, sentence1: str, sentence2: str) -> int:
    """预测单对句子的语义等价性，返回 0（不等价）或 1（等价）"""
    SYSTEM_PROMPT = "你是一个文本语义相似度判断助手。判断两个句子是否语义等价（表达相同的意图）。"
    USER_TEMPLATE = "句子1：{sentence1}\n句子2：{sentence2}\n\n请判断以上两个句子是否语义等价，只回答\"等价\"或\"不等价\"。"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(sentence1=sentence1, sentence2=sentence2)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = outputs[0][len(inputs.input_ids[0]):]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return 1 if ("等价" in gen_text and "不等价" not in gen_text) else 0


def run_evaluation(model, tokenizer, df: pd.DataFrame):
    """批量推理 + 计算指标"""
    log(f"\n开始推理 ({len(df)} 条)...")
    start = time.time()

    y_true = []
    y_pred = []
    errors = []

    for i, row in df.iterrows():
        s1, s2, label = str(row["sentence1"]), str(row["sentence2"]), int(row["label"])
        pred = predict(model, tokenizer, s1, s2)

        y_true.append(label)
        y_pred.append(pred)

        if label != pred:
            errors.append((s1, s2, label, pred))

    elapsed = time.time() - start
    log(f"推理完成，耗时: {elapsed:.1f}s (平均 {elapsed/len(df)*1000:.0f}ms/条)")

    return y_true, y_pred, errors


def compute_metrics(y_true, y_pred):
    """计算分类指标"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1


def main():
    log("=" * 55)
    log("SFT + LoRA 语义相似度 — 推理评估")
    log("=" * 55)

    # 1. 加载模型
    model, tokenizer = load_model()

    # 2. 加载数据
    df = load_test_data(n_samples=500)

    # 3. 推理
    y_true, y_pred, errors = run_evaluation(model, tokenizer, df)

    # 4. 计算指标
    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

    log("\n" + "=" * 55)
    log("评估结果")
    log("=" * 55)
    log(f"Accuracy:  {accuracy:.4f}")
    log(f"Precision: {precision:.4f}")
    log(f"Recall:    {recall:.4f}")
    log(f"F1 Score:  {f1:.4f}")
    log(f"总样本:    {len(y_true)}")
    log(f"正确:      {sum(1 for t, p in zip(y_true, y_pred) if t == p)}")
    log(f"错误:      {len(errors)}")

    # 5. 错误案例
    if errors:
        log(f"\n--- 错误预测示例 (前 10 条) ---")
        for s1, s2, true_l, pred_l in errors[:10]:
            true_text = "等价" if true_l == 1 else "不等价"
            pred_text = "等价" if pred_l == 1 else "不等价"
            log(f"  真实:{true_text} → 预测:{pred_text}")
            log(f"    句1: {s1[:60]}")
            log(f"    句2: {s2[:60]}")
            log("")

    log("评估完成！")


if __name__ == "__main__":
    main()
