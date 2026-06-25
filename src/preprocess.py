"""
数据预处理：将 BQ Corpus 的 sentence pair + label 转换为 SFT 格式
生成训练/验证/测试集，保存为 JSON 供训练使用
"""

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from config import (
    TRAIN_FILE, DEV_FILE, DATA_DIR, DATA_CONFIG,
    SYSTEM_PROMPT, USER_TEMPLATE, LABEL_MAP,
    CHATML_TOKENS,
)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载原始 CSV 数据"""
    train_df = pd.read_csv(TRAIN_FILE)
    dev_df = pd.read_csv(DEV_FILE)
    print(f"训练集原始大小: {len(train_df)}")
    print(f"验证集原始大小: {len(dev_df)}")
    print(f"正样本比例(训练): {train_df['label'].mean():.2%}")
    print(f"正样本比例(验证): {dev_df['label'].mean():.2%}")
    return train_df, dev_df


def format_sft_sample(sentence1: str, sentence2: str, label: int) -> dict:
    """
    将一条样本格式化为 ChatML SFT 格式

    Args:
        sentence1: 第一个句子
        sentence2: 第二个句子
        label: 0（不等价）或 1（等价）

    Returns:
        {"messages": [{"role": "system", "content": ...}, ...]}
        兼容 HuggingFace tokenizer.apply_chat_template
    """
    label_text = LABEL_MAP[label]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                sentence1=sentence1,
                sentence2=sentence2,
            ),
        },
        {"role": "assistant", "content": label_text},
    ]
    return {"messages": messages}


def prepare_dataset() -> DatasetDict:
    """
    完整数据预处理流水线：
    1. 加载原始数据
    2. 从训练集中划分测试集
    3. 转换为 SFT 格式
    4. 构建 HuggingFace DatasetDict
    """
    train_df, dev_df = load_raw_data()

    # 从训练集划出 5% 作为测试集
    train_df, test_df = train_test_split(
        train_df,
        test_size=DATA_CONFIG["test_split_ratio"],
        random_state=DATA_CONFIG["random_seed"],
        stratify=train_df["label"],  # 分层采样，保持正负比例
    )
    print(f"划分后 - 训练集: {len(train_df)}, 验证集: {len(dev_df)}, 测试集: {len(test_df)}")

    # 转换为 SFT 消息格式
    def convert(df: pd.DataFrame):
        records = []
        for _, row in df.iterrows():
            record = format_sft_sample(
                sentence1=str(row["sentence1"]),
                sentence2=str(row["sentence2"]),
                label=int(row["label"]),
            )
            records.append(record)
        return Dataset.from_list(records)

    dataset = DatasetDict({
        "train": convert(train_df),
        "validation": convert(dev_df),
        "test": convert(test_df),
    })

    # 保存处理后的数据（可选，便于检查格式）
    save_path = os.path.join(DATA_DIR, "processed")
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"处理后数据集已保存至: {save_path}")

    # 打印一条样本示例
    print("\n--- SFT 格式示例 (train[0]) ---")
    sample = dataset["train"][0]
    for msg in sample["messages"]:
        print(f"[{msg['role']}]: {msg['content'][:80]}...")

    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer):
    """
    将 messages 列表 tokenize 为模型输入

    关键：只在 assistant 回复部分计算 loss（mask out system/user 部分）
    使用 tokenizer.apply_chat_template + 后续处理
    """
    def tokenize_fn(examples: dict):
        """对一批样本进行 tokenize"""
        all_messages = examples["messages"]

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for messages in all_messages:
            # 1. 用 chat template 生成完整文本，得到 input_ids
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 不添加 generation prompt，因为有 assistant 回复
            )

            # 2. Tokenize 完整文本
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=DATA_CONFIG["max_length"],
                padding=False,  # 在 collator 中 padding
                return_tensors=None,
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # 3. 创建 labels：初始化为 input_ids 的副本，然后 mask 掉非 assistant 部分
            labels = input_ids.copy()

            # 构建只有 assistant 回复部分的模板，找到 assistant 回复的开始位置
            assistant_only = tokenizer.apply_chat_template(
                [messages[-1]],  # 只有 assistant 消息
                tokenize=False,
                add_generation_prompt=False,
            )

            # 找到 assistant 回复在完整文本中的位置
            # assistant_only 以 "<|im_start|>assistant\n..." 开头
            # 在完整文本中找到 assistant 部分的起始
            assistant_keyword = "<|im_start|>assistant\n"
            text_before_assistant = text.split(assistant_keyword)[0] + assistant_keyword

            assistant_prefix_ids = tokenizer(
                text_before_assistant,
                truncation=True,
                max_length=DATA_CONFIG["max_length"],
                return_tensors=None,
            )["input_ids"]

            # Mask 掉 assistant 之前的所有 token（设为 -100）
            assistant_start = len(assistant_prefix_ids)
            for i in range(assistant_start):
                labels[i] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    return tokenized


if __name__ == "__main__":
    # 测试：运行预处理并打印统计信息
    from transformers import AutoTokenizer
    from config import MODEL_PATH

    dataset = prepare_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    print(f"\nTokenized 训练集大小: {len(tokenized_dataset['train'])}")
    print(f"Tokenized 验证集大小: {len(tokenized_dataset['validation'])}")
    print(f"Tokenized 测试集大小: {len(tokenized_dataset['test'])}")

    # 检查一条样本的 token 长度
    sample = tokenized_dataset["train"][0]
    print(f"样本 input_ids 长度: {len(sample['input_ids'])}")
    print(f"有效 labels 数量（非 -100）: {sum(1 for l in sample['labels'] if l != -100)}")
