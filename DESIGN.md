# Qwen2.5-0.5B 文本语义相似度匹配微调方案

## 1. 方案概述

### 1.1 任务定义
给定两个中文句子（sentence1, sentence2），判断它们是否**语义等价**（表达相同意图）。这是一个二分类任务，使用 BQ Corpus（银行客服领域）进行 SFT + LoRA 微调。

### 1.2 技术选型理由

| 选择 | 理由 |
|------|------|
| **Qwen2.5-0.5B-Instruct** | 原生支持中文；Instruct 版本已对齐指令格式，SFT 起点更好；0.5B 规模适合单卡训练 |
| **SFT（指令微调）** | 将分类任务转化为生成任务，利用模型的自然语言理解能力，比直接加分类头更通用 |
| **LoRA** | 0.5B 全量微调需 ~2GB 显存，LoRA 仅训练 ~0.3% 参数（约 1.5M），显存 < 6GB，训练速度快 3-5x，适配器文件仅 ~6MB |

### 1.3 SFT 任务格式设计

将二分类任务转化为**指令遵循**的生成任务：

```
<|im_start|>system
你是一个文本语义相似度判断助手。判断两个句子是否语义等价（表达相同的意图）。
<|im_end|>
<|im_start|>user
句子1：{sentence1}
句子2：{sentence2}

请判断以上两个句子是否语义等价，只回答"等价"或"不等价"。
<|im_end|>
<|im_start|>assistant
等价
<|im_end|>
```

**设计要点**：
- 使用 Qwen2.5 的 ChatML 格式（`<|im_start|>` / `<|im_end|>`），与预训练一致
- 输出简化为"等价"/"不等价"，降低 token 长度，加速收敛
- System prompt 固定任务描述，让模型建立任务上下文

## 2. 模型架构

```
Qwen2.5-0.5B-Instruct (frozen)
    │
    ├── LoRA Adapters (trainable)
    │   ├── q_proj  (LoRA rank=8)
    │   ├── k_proj  (LoRA rank=8)
    │   ├── v_proj  (LoRA rank=8)
    │   ├── o_proj  (LoRA rank=8)
    │   ├── gate_proj (LoRA rank=8)
    │   ├── up_proj   (LoRA rank=8)
    │   └── down_proj (LoRA rank=8)
    │
    └── LM Head (frozen, shared with embeddings)
```

### LoRA 参数
- **rank (r)**: 8 — 语义相似度任务相对简单，rank=8 足够捕捉任务特征
- **alpha**: 16 — alpha = 2*r，缩放因子为 alpha/r = 2
- **dropout**: 0.1 — 轻微正则化防止过拟合
- **target_modules**: 全部 attention 线性层 + FFN 线性层（覆盖最核心的参数矩阵）

### 参数量估算
- Qwen2.5-0.5B 总参数：~494M
- LoRA 可训练参数：~1.5M（约 0.3%）
- 预计显存占用：~4-6GB（batch_size=8, max_length=256）

## 3. 数据策略

### 3.1 数据集划分
| 集合 | 样本数 | 用途 |
|------|--------|------|
| 训练集 | 100,000 | SFT 训练 |
| 验证集 | 10,000 | 每 epoch 评估 + 早停 |
| 测试集 | 从训练集采样 5,000 | 最终评估报告 |

### 3.2 数据分布检查
训练前检查正负样本比例，若不均衡则采用加权损失或在 prompt 中平衡采样。

### 3.3 数据预处理
- 截断：max_length=256 tokens（BQ 问题对通常 < 100 字符）
- 只在 assistant 回答部分计算 loss（mask out system/user tokens）
- Prompt 固定部分使用 `tokenizer.apply_chat_template`

## 4. 训练配置

| 超参数 | 值 | 理由 |
|--------|-----|------|
| **优化器** | AdamW | Transformer 微调标准选择 |
| **学习率** | 2e-4 | LoRA 推荐比全量微调高 10x |
| **学习率调度** | cosine + warmup | 稳定训练，避免初期震荡 |
| **warmup_ratio** | 0.1 | 前 10% 步线性升温 |
| **batch_size** | 8 (per device) × 2 (grad accum) = 16 effective | 适配 6GB 显存 |
| **epochs** | 3 | 小模型 + LoRA，3 轮足够收敛 |
| **weight_decay** | 1e-4 | 轻量正则化 |
| **max_length** | 256 | 覆盖 99%+ 的样本 |
| **fp16/bf16** | bf16（若 GPU 支持）否则 fp16 | 混合精度加速训练 |
| **gradient_checkpointing** | True | 进一步节省显存 |

### 早停策略
- 监控指标：验证集 accuracy
- patience：2 个评估步（约半个 epoch）
- 评估频率：每 2000 步

## 5. 评估方案

### 5.1 主要指标
| 指标 | 含义 |
|------|------|
| **Accuracy** | 整体正确率 |
| **Precision** | 预测为"等价"的精确率 |
| **Recall** | 真实"等价"的召回率 |
| **F1 Score** | 精确率与召回率的调和平均 |
| **AUC-ROC** | 排序能力（对"等价"和"不等价"的区分度） |

### 5.2 评估方式
- **生成式评估**：模型生成文本 → 解析"等价"/"不等价" → 与 label 比较
- 对无法解析的输出（模型生成错误格式），归为预测错误

### 5.3 对比基线
- **BERT-base-chinese + 分类头**（已下载在 models/bert-base-chinese）：经典句对分类方案
- **Qwen2.5-0.5B-Instruct zero-shot**：直接问模型判断（不做微调）

## 6. 推理部署

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "outputs/lora")
tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B-Instruct")

# 推理时控制生成长度，只输出 2-3 个 token
outputs = model.generate(..., max_new_tokens=5)
```

## 7. 文件结构

```
NLPSFTProject/
├── DESIGN.md                  # 本方案文档
├── requirements.txt           # 依赖包
├── src/
│   ├── config.py              # 所有配置常量
│   ├── preprocess.py          # 数据加载与 SFT 格式转换
│   ├── train.py               # 训练主脚本
│   └── inference.py           # 推理与评估脚本
├── outputs/
│   └── lora/                  # LoRA 适配器保存目录
└── logs/                      # 训练日志
```

## 8. 风险与应对

| 风险 | 应对措施 |
|------|----------|
| 模型输出格式不稳定（生成多余文字） | 1) 训练时严格控制输出格式 2) 推理时用正则提取 3) 考虑约束解码 |
| 过拟合（数据量相对小） | LoRA dropout + weight_decay + 早停 |
| 正负样本不均衡 | 检查分布，必要时加权采样或 focal loss |
| 银行领域术语泛化差 | 这是领域内任务，BQ Corpus 覆盖面足够 |
