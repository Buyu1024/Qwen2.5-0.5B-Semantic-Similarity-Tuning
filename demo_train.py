"""
训练 Demo：SFT + LoRA 微调 Qwen2.5-0.5B 做语义相似度匹配
针对 RTX 5090 32GB (Blackwell, CUDA 13.0) 做极致性能优化

优化清单：
  1. Flash Attention 2       → 注意力计算 2-5x 加速，省显存
  2. torch.compile           → 图编译优化 20-40% 吞吐提升
  3. Fused AdamW             → 融合优化器 kernel，减少内存访问
  4. tf32 + cudnn benchmark  → Ampere+ 张量核心加速
  5. 大批量训练 (bs=32+)     → 充分利用 32GB 显存
  6. pin_memory + 多 worker  → 数据加载异步化，消除 CPU 瓶颈
  7. SDPA (fallback)         → Flash Attn 不可用时自动降级

运行：  uv run python demo_train.py [--benchmark]
       uv run python demo_train.py --full    # 完整 3 epoch 训练
"""

import os
import sys
import math
import time
import json
import argparse
import warnings
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-0.5B-Instruct")
TRAIN_FILE = os.path.join(BASE_DIR, "data", "BQ_Corpus", "train.csv")
DEV_FILE   = os.path.join(BASE_DIR, "data", "BQ_Corpus", "dev.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "lora_optimized")

# ============================================================
# 性能配置 — 根据 RTX 5090 32GB 调优
# ============================================================
@dataclass
class PerfConfig:
    # 批次：32GB 显存足够跑大 batch（LoRA 下功耗很低）
    batch_size: int = 24          # per-step batch size
    grad_accum_steps: int = 2     # effective batch = 24 * 2 = 48
    micro_batch_size: int = 8     # 用于 gradient checkpointing 的 micro batch

    # 精度
    use_bf16: bool = True
    allow_tf32: bool = True       # Ampere+ 张量核心，matmul 加速

    # Flash Attention
    use_flash_attn: bool = True   # 优先使用，不可用则回退 SDPA

    # torch.compile
    use_compile: bool = False     # 暂禁（RTX 5090 + PyTorch 2.12 兼容问题）
    compile_mode: str = "default"  # reduce-overhead | max-autotune | default

    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    # 优化器
    use_fused_adamw: bool = True  # 融合 kernel，比标准 AdamW 快 ~15%

    # 训练
    max_length: int = 256
    epochs: int = 1               # demo 默认 1 epoch
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Gradient checkpointing（省显存换计算，大批量时开不开启取决于显存）
    use_grad_ckpt: bool = False   # 32GB 显存不需要，关闭以提速


CFG = PerfConfig()


# ============================================================
# 工具函数
# ============================================================
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def fmt_bytes(b: int) -> str:
    return f"{b / 1024**3:.1f} GB" if b > 1024**3 else f"{b / 1024**2:.0f} MB"

def fmt_time(seconds: float) -> str:
    if seconds < 60:  return f"{seconds:.1f}s"
    if seconds < 3600: return f"{seconds // 60:.0f}m{seconds % 60:.0f}s"
    return f"{seconds // 3600:.0f}h{(seconds % 3600) // 60:.0f}m"


# ============================================================
# Step 1: 环境检测 & 性能调优
# ============================================================
def setup_environment():
    """检测硬件并设置最优性能参数"""
    log("=" * 60)
    log("🔥 RTX 5090 性能优化 — 环境检测")
    log("=" * 60)

    log(f"Python:   {sys.version.split()[0]}")
    log(f"PyTorch:  {torch.__version__}")
    log(f"CUDA:     {torch.version.cuda}")

    if not torch.cuda.is_available():
        log("❌ CUDA 不可用，退出")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory
    gpu_cc   = torch.cuda.get_device_properties(0).major + \
               torch.cuda.get_device_properties(0).minor / 10.0
    log(f"GPU:      {gpu_name}")
    log(f"显存:     {fmt_bytes(gpu_mem)}")
    log(f"算力:     sm_{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

    # === 1. TF32 加速（Ampere+ 可用）================================
    if CFG.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log("✅ TF32 matmul 加速已开启")

    # === 2. cuDNN benchmark ========================================
    torch.backends.cudnn.benchmark = True
    log("✅ cuDNN benchmark 已开启")

    # === 3. Flash Attention 2 检测 ==================================
    has_flash_attn = False
    if CFG.use_flash_attn:
        try:
            import flash_attn
            has_flash_attn = True
            log(f"✅ Flash Attention 2: {flash_attn.__version__}")
        except ImportError:
            log("⚠️  Flash Attention 2 未安装，回退到 SDPA")
            log("   安装命令: pip install flash-attn --no-build-isolation")

    # === 4. torch.compile 检测 ======================================
    compile_available = CFG.use_compile and hasattr(torch, "compile")
    if compile_available:
        log(f"✅ torch.compile 可用 (mode={CFG.compile_mode})")
    else:
        log("⚠️  torch.compile 不可用（PyTorch < 2.0 或已禁用）")

    # === 5. Fused AdamW ============================================
    fused_available = CFG.use_fused_adamw and torch.cuda.is_available()
    if fused_available:
        log("✅ Fused AdamW 可用")
    CFG.use_fused_adamw = fused_available

    # === 6. 根据显存自动调 batch size ================================
    vram_gb = gpu_mem / 1024**3
    if vram_gb >= 48:
        CFG.batch_size = 40
    elif vram_gb >= 32:
        CFG.batch_size = 24
    elif vram_gb >= 24:
        CFG.batch_size = 16
    elif vram_gb >= 16:
        CFG.batch_size = 8
    else:
        CFG.batch_size = 4
    log(f"📊 自动批次大小: {CFG.batch_size} (适配 {vram_gb:.0f}GB 显存)")

    return {
        "has_flash_attn": has_flash_attn,
        "compile_available": compile_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
    }


# ============================================================
# Step 2: 数据加载
# ============================================================
SYSTEM_PROMPT = "你是一个文本语义相似度判断助手。判断两个句子是否语义等价（表达相同的意图）。"
USER_TEMPLATE = '''句子1：{sentence1}
句子2：{sentence2}

请判断以上两个句子是否语义等价，只回答"等价"或"不等价"。'''
LABEL_MAP = {0: "不等价", 1: "等价"}


def load_and_prepare_data(n_train: int = 500, n_dev: int = 200):
    """加载并准备小样本数据"""
    log("\n" + "=" * 60)
    log("📦 Step 2: 数据加载")
    log("=" * 60)

    train_df = pd.read_csv(TRAIN_FILE).sample(n=n_train, random_state=42)
    dev_df   = pd.read_csv(DEV_FILE).sample(n=n_dev, random_state=42)

    train_df = train_df.reset_index(drop=True)
    dev_df   = dev_df.reset_index(drop=True)

    log(f"训练集: {len(train_df)} 条 (正样本 {train_df['label'].mean():.1%})")
    log(f"验证集: {len(dev_df)}   条 (正样本 {dev_df['label'].mean():.1%})")

    # 构建 messages
    def df_to_messages(df):
        records = []
        for _, row in df.iterrows():
            records.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    sentence1=str(row["sentence1"]),
                    sentence2=str(row["sentence2"]),
                )},
                {"role": "assistant", "content": LABEL_MAP[int(row["label"])]},
            ])
        return records

    return df_to_messages(train_df), df_to_messages(dev_df), train_df, dev_df


def tokenize_messages(tokenizer, messages_list, max_length=256):
    """Tokenize 并构建 loss mask"""
    input_ids_list, attention_mask_list, labels_list = [], [], []
    assistant_marker = "<|im_start|>assistant\n"

    for messages in messages_list:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        input_ids = tokenized["input_ids"]
        attn_mask = tokenized["attention_mask"]
        labels    = input_ids.copy()

        # Mask: 只对 assistant 回复部分计算 loss
        if assistant_marker in full_text:
            prefix = full_text.split(assistant_marker)[0] + assistant_marker
            assist_start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            for i in range(min(assist_start, len(labels))):
                labels[i] = -100
        else:
            for i in range(len(labels) - 5):
                labels[i] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attn_mask)
        labels_list.append(labels)

    return input_ids_list, attention_mask_list, labels_list


# ============================================================
# Step 3: 加载模型 + 优化
# ============================================================
def load_model_with_optimizations(env_info: dict):
    """加载 Qwen2.5-0.5B，注入 LoRA，应用全部优化"""
    log("\n" + "=" * 60)
    log("🤖 Step 3: 加载模型 & 注入 LoRA & 编译优化")
    log("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # --- Tokenizer ---
    log("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  vocab_size: {len(tokenizer)}, pad_token: {tokenizer.pad_token}")

    # --- 选择 attention 实现 ---
    attn_impl = "sdpa"  # default
    if env_info["has_flash_attn"]:
        attn_impl = "flash_attention_2"
        log("🚀 使用 Flash Attention 2")
    else:
        log("📌 使用 PyTorch SDPA（scaled_dot_product_attention）")

    # --- 加载模型 ---
    log("加载基础模型...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if CFG.use_bf16 else torch.float16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    log(f"  加载耗时: {time.time() - t0:.1f}s")
    log(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # --- LoRA 注入 ---
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Gradient Checkpointing ---
    if CFG.use_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        log("📌 Gradient Checkpointing: 开启（节省显存）")
    else:
        model.config.use_cache = False  # 训练时必须关闭
        log("📌 Gradient Checkpointing: 关闭（满速训练）")

    # --- torch.compile ---
    if env_info["compile_available"]:
        log(f"🔥 torch.compile (mode={CFG.compile_mode}) 编译中...")
        t0 = time.time()
        try:
            model = torch.compile(model, mode=CFG.compile_mode)
            log(f"  编译耗时: {time.time() - t0:.1f}s")
        except Exception as e:
            log(f"  ⚠️ 编译失败: {e}，回退到 eager 模式")
            env_info["compile_available"] = False

    return model, tokenizer


# ============================================================
# Step 4: DataLoader（高性能）
# ============================================================
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attn_mask, labels):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.labels    = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch, pad_token_id):
    """动态 padding + pre-padding 缓存优化"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    n = len(batch)

    input_ids     = torch.full((n, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)
    labels         = torch.full((n, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        sl = item["input_ids"].size(0)
        input_ids[i, :sl]     = item["input_ids"]
        attention_mask[i, :sl] = item["attention_mask"]
        labels[i, :sl]         = item["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def create_dataloaders(train_data, dev_data, tokenizer):
    """创建性能优化的 DataLoader"""
    log("\n" + "=" * 60)
    log("⚡ Step 4: DataLoader（高性能配置）")
    log("=" * 60)

    train_ds = SFTDataset(*train_data)
    dev_ds   = SFTDataset(*dev_data)

    from functools import partial
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        prefetch_factor=CFG.prefetch_factor if CFG.num_workers > 0 else None,
        persistent_workers=CFG.num_workers > 0,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_ds,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        prefetch_factor=CFG.prefetch_factor if CFG.num_workers > 0 else None,
        persistent_workers=CFG.num_workers > 0,
    )

    log(f"训练批次: {len(train_loader)} × bs={CFG.batch_size} × ga={CFG.grad_accum_steps}")
    log(f"  有效 batch size: {CFG.batch_size * CFG.grad_accum_steps}")
    log(f"验证批次: {len(dev_loader)} × bs={CFG.batch_size * 2}")
    log(f"DataLoader workers: {CFG.num_workers}, pin_memory: {CFG.pin_memory}")

    return train_loader, dev_loader


# ============================================================
# Step 5: 训练循环（极致优化版）
# ============================================================
def train_epoch(model, train_loader, dev_loader, optimizer, scheduler, env_info):
    """跑一个 epoch 的训练循环"""
    log("\n" + "=" * 60)
    log(f"🏋️ Step 5: 训练 ({CFG.epochs} epoch, bs={CFG.batch_size * CFG.grad_accum_steps})")
    log("=" * 60)

    total_steps   = len(train_loader)
    device        = model.device
    train_losses  = []
    best_dev_loss = float("inf")
    grad_accum    = CFG.grad_accum_steps
    scaler        = None  # bf16 不需要 GradScaler

    model.train()
    optimizer.zero_grad(set_to_none=True)  # set_to_none 比 zero_grad() 更快

    for epoch in range(CFG.epochs):
        epoch_loss = []
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            # --- 数据转移到 GPU ---
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # --- 前向 ---
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if CFG.use_bf16 else torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            # --- 反向 ---
            loss.backward()

            train_losses.append(float(outputs.loss))
            epoch_loss.append(float(outputs.loss))

            # --- 梯度累积步 ---
            if (step + 1) % grad_accum == 0 or (step + 1) == total_steps:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=CFG.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # --- 日志 ---
            log_interval = max(1, total_steps // 10)
            if (step + 1) % log_interval == 0:
                recent_loss = np.mean(train_losses[-log_interval:])
                lr = scheduler.get_last_lr()[0]
                pct = (step + 1) / total_steps * 100
                elapsed = time.time() - epoch_start
                eta = elapsed / (step + 1) * (total_steps - step - 1)
                log(f"  Epoch {epoch+1} | Step {step+1:3d}/{total_steps} ({pct:.0f}%) | "
                    f"Loss: {recent_loss:.4f} | LR: {lr:.2e} | ETA: {fmt_time(eta)}")

        # --- Epoch 结束：验证 ---
        avg_epoch_loss = np.mean(epoch_loss)
        dev_loss = evaluate(model, dev_loader, device)

        log(f"\n  === Epoch {epoch+1} 总结 === ")
        log(f"  Train Loss: {avg_epoch_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
            f"耗时: {fmt_time(time.time() - epoch_start)}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            log(f"  ✅ 最佳验证 loss: {best_dev_loss:.4f}")

    return train_losses, best_dev_loss


@torch.no_grad()
def evaluate(model, dev_loader, device):
    """验证"""
    model.eval()
    losses = []
    for batch in dev_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if CFG.use_bf16 else torch.float16):
            outputs = model(**batch)
            losses.append(float(outputs.loss))
    model.train()
    return np.mean(losses)


# ============================================================
# Step 6: 推理验证
# ============================================================
@torch.no_grad()
def run_inference(model, tokenizer, test_cases):
    """用微调后的模型做推理"""
    log("\n" + "=" * 60)
    log("🧪 Step 6: 推理验证")
    log("=" * 60)

    model.eval()
    model.config.use_cache = True

    correct = 0
    for s1, s2, true_label in test_cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(sentence1=s1, sentence2=s2)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs, max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        pred_label = 1 if ("等价" in gen_text and "不等价" not in gen_text) else 0
        status = "✅" if pred_label == true_label else "❌"
        correct += (pred_label == true_label)

        true_text = "等价" if true_label == 1 else "不等价"
        pred_text = "等价" if pred_label == 1 else "不等价"
        log(f"  {status} 真实:{true_text} → 预测:{pred_text} | 输出: '{gen_text}'")
        log(f"     句1: {s1}")
        log(f"     句2: {s2}")

    log(f"\n  推理准确率: {correct}/{len(test_cases)}")
    return correct


# ============================================================
# Step 7: 性能基准测试（可选）
# ============================================================
def run_benchmark(model, train_loader):
    """测试训练吞吐量 (tokens/sec) 和显存占用"""
    log("\n" + "=" * 60)
    log("📊 性能基准测试")
    log("=" * 60)

    device = model.device
    model.train()

    # 预热
    log("预热中 (5 steps)...")
    for i, batch in enumerate(train_loader):
        if i >= 5: break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(**batch)

    torch.cuda.synchronize()

    # 计时
    log("测速中 (20 steps)...")
    total_tokens = 0
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 20: break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        total_tokens += batch["attention_mask"].sum().item()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(**batch)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    tokens_per_sec = total_tokens / elapsed
    peak_mem = torch.cuda.max_memory_allocated()

    log(f"  吞吐量:   {tokens_per_sec:.0f} tokens/sec")
    log(f"  峰值显存: {fmt_bytes(peak_mem)}")
    log(f"  每步耗时: {elapsed / 20 * 1000:.0f} ms")

    return tokens_per_sec, peak_mem


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="性能基准测试模式")
    parser.add_argument("--full", action="store_true", help="完整 3 epoch 训练")
    parser.add_argument("--n_train", type=int, default=500, help="训练样本数")
    parser.add_argument("--n_dev", type=int, default=200, help="验证样本数")
    parser.add_argument("--no_flash_attn", action="store_true", help="禁用 Flash Attention")
    parser.add_argument("--no_compile", action="store_true", help="禁用 torch.compile")
    args = parser.parse_args()

    if args.full:
        CFG.epochs = 3
    if args.no_flash_attn:
        CFG.use_flash_attn = False
    if args.no_compile:
        CFG.use_compile = False

    # ---- 环境检测 ----
    env_info = setup_environment()

    # ---- 数据准备 ----
    train_msgs, dev_msgs, train_df, dev_df = load_and_prepare_data(args.n_train, args.n_dev)

    # ---- 模型加载 ----
    model, tokenizer = load_model_with_optimizations(env_info)

    # ---- Tokenize ----
    log("\nTokenizing...")
    t0 = time.time()
    train_data = tokenize_messages(tokenizer, train_msgs, max_length=CFG.max_length)
    dev_data   = tokenize_messages(tokenizer, dev_msgs, max_length=CFG.max_length)
    lengths = [len(ids) for ids in train_data[0]]
    log(f"  Tokenize 耗时: {time.time() - t0:.1f}s")
    log(f"  Token 长度: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.0f}")

    # ---- DataLoader ----
    train_loader, dev_loader = create_dataloaders(train_data, dev_data, tokenizer)

    # ---- 基准测试（可选） ----
    if args.benchmark:
        run_benchmark(model, train_loader)
        return

    # ---- 优化器 & 调度器 ----
    log("\n" + "=" * 60)
    log("🔧 优化器 & 调度器")
    log("=" * 60)

    total_steps = len(train_loader) * CFG.epochs
    warmup_steps = max(1, int(total_steps * CFG.warmup_ratio))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        fused=CFG.use_fused_adamw,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )
    # 简化 warmup：前 warmup_steps 线性增长
    def warmup_scheduler(step):
        if step < warmup_steps:
            return CFG.learning_rate * step / warmup_steps
        return scheduler.get_last_lr()[0]

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  优化器: AdamW {'(fused)' if CFG.use_fused_adamw else ''}  |  LR: {CFG.learning_rate}")
    log(f"  调度器: linear warmup ({warmup_steps} steps) + cosine decay")
    log(f"  总步数: {total_steps}  |  可训练参数: {trainable_params:,}")

    # 注意：这里为了兼容 warmup，使用 LambdaLR 包装
    from functools import partial
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            step / max(1, warmup_steps)
            if step < warmup_steps
            else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        ),
    )

    # ---- 训练 ----
    train_start = time.time()
    train_losses, best_dev_loss = train_epoch(
        model, train_loader, dev_loader, optimizer, scheduler, env_info,
    )
    train_time = time.time() - train_start

    # ---- 保存模型 ----
    log("\n" + "=" * 60)
    log("💾 Step 7: 保存 LoRA 适配器")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    saved_files = os.listdir(OUTPUT_DIR)
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in saved_files)
    log(f"  保存路径: {OUTPUT_DIR}")
    log(f"  文件大小: {fmt_bytes(total_size)} ({len(saved_files)} 个文件)")

    # ---- 推理验证 ----
    test_cases = [
        ("微粒咨询电话号码多少", "你们的人工客服电话是多少", 1),
        ("微信消费算吗", "还有多少钱没还", 0),
        ("为什么借款后一直没有给我回拨电话", "怎么申请借款后没有打电话过来呢", 1),
        ("我要关闭微粒贷这个功能", "提示未满足条件", 0),
        ("借款后多长时间给打电话", "借款后多久打电话啊", 1),
    ]
    correct = run_inference(model, tokenizer, test_cases)

    # ---- 总结 ----
    log("\n" + "=" * 60)
    log("🎉 训练 Demo 完成！")
    log("=" * 60)

    # 计算吞吐量
    if train_losses:
        total_tokens_trained = sum(
            batch["attention_mask"].sum().item()
            for i, batch in enumerate(train_loader)
        ) * CFG.epochs
        throughput = total_tokens_trained / train_time

    log(f"  训练耗时:    {fmt_time(train_time)}")
    log(f"  吞吐量:      {throughput:.0f} tokens/sec" if train_losses else "  N/A")
    log(f"  初始 Loss:   {train_losses[0]:.4f}" if train_losses else "  N/A")
    log(f"  最终 Loss:   {np.mean(train_losses[-20:]):.4f}" if train_losses else "  N/A")
    log(f"  最佳 Dev Loss: {best_dev_loss:.4f}")
    log(f"  推理准确率:  {correct}/{len(test_cases)}")
    log(f"  LoRA 大小:   {fmt_bytes(total_size)}")

    log("\n🚀 下一步：")
    log("  python demo_train.py --full          # 完整 3 epoch 训练")
    log("  python demo_train.py --benchmark      # 性能基准测试")
    log("  python src/train.py                  # 全量数据正式训练")


if __name__ == "__main__":
    main()
