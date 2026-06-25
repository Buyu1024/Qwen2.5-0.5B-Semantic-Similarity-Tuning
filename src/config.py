"""
全局配置：模型路径、LoRA参数、训练超参数
"""

import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-0.5B-Instruct")
DATA_DIR = os.path.join(BASE_DIR, "data", "BQ_Corpus")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
DEV_FILE = os.path.join(DATA_DIR, "dev.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "lora")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ============================================================
# 模型配置
# ============================================================
MODEL_NAME = "Qwen2.5-0.5B-Instruct"
TRUST_REMOTE_CODE = True
TORCH_DTYPE = "auto"  # bf16 if available, else fp16

# ============================================================
# LoRA 配置
# ============================================================
LORA_CONFIG = {
    "r": 8,                      # LoRA rank
    "lora_alpha": 16,            # LoRA alpha (scaling = alpha/r = 2)
    "lora_dropout": 0.1,         # LoRA dropout
    "bias": "none",              # 不训练 bias
    "task_type": "CAUSAL_LM",    # 因果语言模型
    "target_modules": [
        "q_proj",                # Query projection
        "k_proj",                # Key projection
        "v_proj",                # Value projection
        "o_proj",                # Output projection
        "gate_proj",             # FFN gate (SwiGLU)
        "up_proj",               # FFN up projection
        "down_proj",             # FFN down projection
    ],
}

# ============================================================
# 训练超参数
# ============================================================
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,       # effective batch = 8 * 2 = 16
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_torch",
    "fp16": False,                          # 优先使用 bf16
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 100,
    "eval_steps": 2000,
    "save_steps": 2000,
    "save_total_limit": 2,                  # 只保留最近2个checkpoint
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_accuracy",
    "greater_is_better": True,
    "dataloader_num_workers": 2,
    "remove_unused_columns": False,
    "report_to": "none",                    # 不用 wandb/tensorboard
}

# ============================================================
# 数据配置
# ============================================================
DATA_CONFIG = {
    "max_length": 256,                      # 最大 token 长度
    "test_split_ratio": 0.05,               # 从训练集划出 5% 做测试集
    "random_seed": 42,
}

# ============================================================
# Prompt 模板（ChatML 格式，适配 Qwen2.5）
# ============================================================
SYSTEM_PROMPT = "你是一个文本语义相似度判断助手。判断两个句子是否语义等价（表达相同的意图）。"

USER_TEMPLATE = """句子1：{sentence1}
句子2：{sentence2}

请判断以上两个句子是否语义等价，只回答"等价"或"不等价"。”"""

# 标签映射
LABEL_MAP = {
    0: "不等价",
    1: "等价",
}

# ChatML 特殊 token
CHATML_TOKENS = {
    "system_start": "<|im_start|>system\n",
    "system_end": "<|im_end|>\n",
    "user_start": "<|im_start|>user\n",
    "user_end": "<|im_end|>\n",
    "assistant_start": "<|im_start|>assistant\n",
    "assistant_end": "<|im_end|>\n",
}
