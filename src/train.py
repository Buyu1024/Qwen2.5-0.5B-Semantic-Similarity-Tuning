"""
SFT + LoRA 正式训练脚本 — RTX 5090 32GB 优化版

性能优化:
  - Flash Attention 2（2-5x 注意力加速）
  - torch.compile（暂禁，RTX 5090 + PyTorch 2.12 兼容性问题）
  - Fused AdamW + TF32 + cuDNN benchmark
  - 大批量训练 + 异步数据加载
  - 混合精度 bf16 + 梯度累积

用法:
  uv run python src/train.py              # 默认配置全量训练
  uv run python src/train.py --test-mode  # 小样本快速测试
  uv run python src/train.py --resume     # 断点续训
"""

import os
import sys
import math
import time
import json
import argparse
import warnings
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-0.5B-Instruct")
DATA_DIR   = os.path.join(BASE_DIR, "data", "BQ_Corpus")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
DEV_FILE   = os.path.join(DATA_DIR, "dev.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "lora")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 训练配置 — RTX 5090 32GB 优化
# ============================================================
TRAINING_CONFIG = {
    # 批次（32GB 显存足够跑大 batch）
    "per_device_train_batch_size": 24,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 2,     # effective = 24 * 2 = 48

    # 学习率
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",

    # 训练轮数
    "num_train_epochs": 3,

    # 精度（RTX 5090 原生 bf16）
    "fp16": False,
    "bf16": True,

    # 优化器
    "optim": "adamw_torch_fused",         # Fused AdamW, ~15% faster

    # Gradient checkpointing（32GB 显存够用，关闭以提速）
    "gradient_checkpointing": False,

    # 日志 & 保存
    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 1000,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,

    # 数据加载
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 4,

    # 其他
    "remove_unused_columns": False,
    "report_to": "none",
    "max_grad_norm": 1.0,
}

# LoRA 配置
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

# 数据配置
DATA_CONFIG = {
    "max_length": 256,
    "random_seed": 42,
}

# ============================================================
# Prompt 模板
# ============================================================
SYSTEM_PROMPT = "你是一个文本语义相似度判断助手。判断两个句子是否语义等价（表达相同的意图）。"
USER_TEMPLATE = """句子1：{sentence1}
句子2：{sentence2}

请判断以上两个句子是否语义等价，只回答"等价"或"不等价"。"""
LABEL_MAP = {0: "不等价", 1: "等价"}


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================================================
# 环境优化
# ============================================================
def setup_environment():
    """配置硬件相关优化"""
    if not torch.cuda.is_available():
        log("❌ CUDA 不可用！")
        return {}

    gpu_name    = torch.cuda.get_device_name(0)
    gpu_mem_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    sm_major    = torch.cuda.get_device_properties(0).major
    sm_minor    = torch.cuda.get_device_properties(0).minor

    log("=" * 60)
    log("🔥 环境检测 & 优化配置")
    log("=" * 60)
    log(f"GPU:      {gpu_name}")
    log(f"显存:     {gpu_mem_gb:.1f} GB")
    log(f"算力:     sm_{sm_major}.{sm_minor}")
    log(f"PyTorch:  {torch.__version__}")
    log(f"CUDA:     {torch.version.cuda}")

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log("✅ TF32 matmul 加速")

    # cuDNN
    torch.backends.cudnn.benchmark = True
    log("✅ cuDNN benchmark")

    # Flash Attention 2
    has_fa2 = False
    try:
        import flash_attn
        has_fa2 = True
        log(f"✅ Flash Attention 2: {flash_attn.__version__}")
    except ImportError:
        log("⚠️  Flash Attention 2 未安装，使用 SDPA")
        log("   安装: pip install flash-attn --no-build-isolation")

    # torch.compile
    has_compile = hasattr(torch, "compile")
    if has_compile:
        log("✅ torch.compile 可用")
    else:
        log("⚠️  torch.compile 不可用")

    # 根据显存微调 batch size
    if gpu_mem_gb >= 48:
        TRAINING_CONFIG["per_device_train_batch_size"] = 40
    elif gpu_mem_gb >= 32:
        TRAINING_CONFIG["per_device_train_batch_size"] = 24
    elif gpu_mem_gb >= 24:
        TRAINING_CONFIG["per_device_train_batch_size"] = 16

    log(f"📊 batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
    log(f"📊 effective:  {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")

    return {"has_fa2": has_fa2, "has_compile": has_compile}


# ============================================================
# 数据预处理
# ============================================================
def prepare_dataset(test_mode: bool = False) -> DatasetDict:
    """构建 SFT 格式的 HuggingFace Dataset"""
    log("\n" + "=" * 60)
    log("📦 数据预处理")
    log("=" * 60)

    train_df = pd.read_csv(TRAIN_FILE)
    dev_df   = pd.read_csv(DEV_FILE)

    if test_mode:
        train_df = train_df.sample(n=1000, random_state=DATA_CONFIG["random_seed"])
        dev_df   = dev_df.sample(n=200, random_state=DATA_CONFIG["random_seed"])
        log(f"⚠️  测试模式: train={len(train_df)}, dev={len(dev_df)}")
    else:
        log(f"全量模式: train={len(train_df)}, dev={len(dev_df)}")

    log(f"正样本比例 - train: {train_df['label'].mean():.1%}, dev: {dev_df['label'].mean():.1%}")

    def build_messages(df: pd.DataFrame) -> list:
        records = []
        for _, row in df.iterrows():
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        sentence1=str(row["sentence1"]),
                        sentence2=str(row["sentence2"]),
                    )},
                    {"role": "assistant", "content": LABEL_MAP[int(row["label"])]},
                ],
                "label": int(row["label"]),
            })
        return records

    dataset = DatasetDict({
        "train":      Dataset.from_list(build_messages(train_df)),
        "validation": Dataset.from_list(build_messages(dev_df)),
    })

    log(f"  train:      {len(dataset['train'])}")
    log(f"  validation: {len(dataset['validation'])}")

    # 打印示例
    sample = dataset["train"][0]
    for msg in sample["messages"]:
        log(f"  [{msg['role']}]: {msg['content'][:80]}...")

    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer):
    """Tokenize 并构建 loss mask"""
    log("\nTokenizing...")

    def tokenize_fn(examples):
        all_input_ids, all_attn_mask, all_labels = [], [], []

        for messages in examples["messages"]:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            tok = tokenizer(full_text, truncation=True,
                           max_length=DATA_CONFIG["max_length"], padding=False)

            input_ids = tok["input_ids"]
            labels    = input_ids.copy()

            # Mask: 只在 assistant 部分计算 loss
            marker = "<|im_start|>assistant\n"
            if marker in full_text:
                prefix = full_text.split(marker)[0] + marker
                assist_start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
                for i in range(min(assist_start, len(labels))):
                    labels[i] = -100

            all_input_ids.append(input_ids)
            all_attn_mask.append(tok["attention_mask"])
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attn_mask,
            "labels": all_labels,
        }

    tokenized = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=["messages", "label"],
        desc="Tokenizing",
    )

    # 统计 token 长度
    lengths = [len(ids) for ids in tokenized["train"]["input_ids"]]
    log(f"  Token 长度: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.0f}")

    return tokenized


# ============================================================
# 评估指标
# ============================================================
def compute_metrics(eval_pred, tokenizer):
    """解析生成文本，计算分类指标"""
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    true_labels, pred_labels = [], []

    for pred_seq, label_seq in zip(predictions, labels):
        valid = label_seq != -100
        if not valid.any():
            continue

        true_ids = label_seq[valid]
        true_text = tokenizer.decode(true_ids, skip_special_tokens=True).strip()
        true_labels.append(1 if "等价" in true_text else 0)

        pred_ids = pred_seq[valid]
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        pred_labels.append(1 if "等价" in pred_text else 0)

    if not true_labels:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    return {
        "accuracy":  float((y_true == y_pred).mean()),
        "precision": float(tp / (tp + fp + 1e-8)),
        "recall":    float(tp / (tp + fn + 1e-8)),
        "f1":        float(2 * tp / (2 * tp + fp + fn + 1e-8)),
    }


class MetricsLogger(transformers.TrainerCallback):
    """训练进度日志"""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log(f"\n--- Eval @ step {state.global_step} ---")
            for k in ["eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]:
                if k in metrics:
                    log(f"  {k}: {metrics[k]:.4f}")
            log("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = logs.get("step", state.global_step)
            loss = logs["loss"]
            lr   = logs.get("learning_rate", 0)
            log(f"  Step {step:5d} | loss={loss:.4f} | lr={lr:.2e}")


# ============================================================
# 主训练函数
# ============================================================
def train(test_mode: bool = False, resume: bool = False):
    """训练主流程"""
    log("=" * 60)
    log("🚀 Qwen2.5-0.5B SFT + LoRA 语义相似度微调")
    log("   RTX 5090 32GB 优化版")
    log("=" * 60)

    # ---- 环境 ----
    env = setup_environment()

    # ---- 数据 ----
    dataset = prepare_dataset(test_mode=test_mode)

    # ---- Tokenizer ----
    log("\n加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  vocab: {len(tokenizer)}, pad: {tokenizer.pad_token}")

    # ---- 模型 ----
    log("\n加载模型...")
    attn_impl = "flash_attention_2" if env.get("has_fa2") else "sdpa"
    log(f"  Attention: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    log(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ---- LoRA ----
    log("\n注入 LoRA...")
    lora_config = LoraConfig(**LORA_CONFIG)
    if resume and os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        log("  从 checkpoint 恢复 LoRA...")
        model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- torch.compile ----
    if False:  # torch.compile 在 RTX 5090 + PyTorch 2.12.1 上暂不稳定
        log("\n🔥 torch.compile (max-autotune)...")
        t0 = time.time()
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log(f"  编译耗时: {time.time() - t0:.1f}s")
        except Exception as e:
            log(f"  ⚠️ 编译失败: {e}")

    # ---- Tokenize ----
    tokenized = tokenize_dataset(dataset, tokenizer)

    # ---- 计算总步数 ----
    effective_bs = TRAINING_CONFIG["per_device_train_batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"]
    total_steps  = math.ceil(len(tokenized["train"]) / effective_bs) * TRAINING_CONFIG["num_train_epochs"]
    log(f"\n📊 训练规模:")
    log(f"  训练样本: {len(tokenized['train'])}")
    log(f"  验证样本: {len(tokenized['validation'])}")
    log(f"  有效 batch: {effective_bs}")
    log(f"  总步数: ~{total_steps}")

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # 批次
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        # 训练
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        optim=TRAINING_CONFIG["optim"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        # 精度
        fp16=TRAINING_CONFIG["fp16"],
        bf16=TRAINING_CONFIG["bf16"],
        # Gradient Checkpointing
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        # 日志 & 保存
        logging_dir=LOG_DIR,
        logging_steps=TRAINING_CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
        greater_is_better=TRAINING_CONFIG["greater_is_better"],
        # 数据加载
        dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
        dataloader_pin_memory=TRAINING_CONFIG["dataloader_pin_memory"],
        dataloader_prefetch_factor=TRAINING_CONFIG["dataloader_prefetch_factor"],
        # 其他
        remove_unused_columns=TRAINING_CONFIG["remove_unused_columns"],
        report_to=TRAINING_CONFIG["report_to"],
        seed=DATA_CONFIG["random_seed"],
    )

    # ---- Data Collator ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ---- Trainer ----
    def wrapped_metrics(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=wrapped_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            ),
            MetricsLogger(),
        ],
    )

    # ---- 训练 ----
    log("\n" + "=" * 60)
    log("🏋️ 开始训练")
    log("=" * 60)

    train_start = time.time()

    if resume and os.path.exists(os.path.join(OUTPUT_DIR, "checkpoint-*")):
        checkpoint = sorted([
            f for f in os.listdir(OUTPUT_DIR) if f.startswith("checkpoint-")
        ])[-1]
        log(f"从 checkpoint 恢复: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=os.path.join(OUTPUT_DIR, checkpoint))
    else:
        train_result = trainer.train()

    train_time = time.time() - train_start
    log(f"\n训练完成！总耗时: {train_time / 60:.1f} 分钟")

    # ---- 保存 ----
    log("\n保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ---- 最终评估 ----
    log("\n最终评估...")
    eval_results = trainer.evaluate()
    log(f"验证集指标:")
    for k, v in sorted(eval_results.items()):
        if isinstance(v, (int, float)):
            log(f"  {k}: {v:.4f}")

    # ---- 保存指标 ----
    results = {
        "train_runtime_seconds": train_time,
        "train_steps": train_result.global_step,
        "train_loss": float(train_result.training_loss) if train_result.training_loss else None,
        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
           for k, v in eval_results.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log(f"\n模型 & 指标已保存至: {OUTPUT_DIR}")
    log("🎉 训练完成！")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B SFT+LoRA 训练")
    parser.add_argument("--test-mode", action="store_true",
                       help="小样本快速测试（1000条）")
    parser.add_argument("--resume", action="store_true",
                       help="从 checkpoint 断点续训")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="覆盖默认 batch size")
    args = parser.parse_args()

    if args.batch_size:
        TRAINING_CONFIG["per_device_train_batch_size"] = args.batch_size
        log(f"📊 手动 batch size: {args.batch_size}")

    train(test_mode=args.test_mode, resume=args.resume)
