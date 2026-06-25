#!/bin/bash
# =============================================================
# 在 AutoDL 实例上运行，从 HuggingFace 下载 Qwen2.5-0.5B-Instruct
# 用法: bash download_model.sh
# =============================================================
set -e

MODEL_DIR="models/Qwen2.5-0.5B-Instruct"
HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

echo "============================================"
echo " 下载 Qwen2.5-0.5B-Instruct"
echo " 目标: ${MODEL_DIR}"
echo "============================================"

# 方式1: 用 huggingface_hub 的 snapshot_download（推荐，支持断点续传）
echo ""
echo ">>> 下载模型文件 (~1GB)..."
python3 -c "
from huggingface_hub import snapshot_download
import os

model_dir = '${MODEL_DIR}'
os.makedirs(model_dir, exist_ok=True)

# snapshot_download 会自动跳过已存在的文件
snapshot_download(
    '${HF_MODEL}',
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    ignore_patterns=['.cache', '.cache/*', '*.msgpack', '*.h5'],
)
print(f'✅ 模型已下载到 {model_dir}')
"

# 验证
echo ""
echo ">>> 验证模型文件..."
python3 -c "
import os
model_dir = '${MODEL_DIR}'
files = os.listdir(model_dir)
total = sum(os.path.getsize(os.path.join(model_dir, f)) for f in files)
print(f'  文件数: {len(files)}')
print(f'  总大小: {total/1024**3:.2f} GB')
for f in sorted(files):
    size = os.path.getsize(os.path.join(model_dir, f))
    print(f'  {f:30s} {size/1024**2:6.1f} MB')

# 检查关键文件
required = ['config.json', 'tokenizer_config.json', 'model.safetensors']
missing = [f for f in required if f not in files]
if missing:
    print(f'  ⚠️ 缺少文件: {missing}')
else:
    print(f'  ✅ 关键文件齐全')
"

echo ""
echo "============================================"
echo " 模型下载完成！现在可以开始训练了："
echo "   python3 demo_train.py"
echo "============================================"
