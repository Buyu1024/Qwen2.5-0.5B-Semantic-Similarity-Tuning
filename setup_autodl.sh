#!/bin/bash
# =============================================================
# AutoDL RTX 5090 一键环境初始化脚本
# 用法: bash setup_autodl.sh
# =============================================================
set -e

echo "============================================"
echo " AutoDL RTX 5090 训练环境初始化"
echo "============================================"

# ---- 1. 系统检查 ----
echo ""
echo ">>> 系统信息"
echo "  OS:     $(lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1)"
echo "  Python: $(python3 --version)"
echo "  CUDA:   $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'check with nvidia-smi')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  GPU:    unknown"

# ---- 2. 安装 pip 包 ----
echo ""
echo ">>> 安装 PyTorch 生态包..."
pip install --upgrade pip -q

# 核心包
pip install transformers datasets peft accelerate \
            pandas scikit-learn numpy -q

# ---- 3. Flash Attention 2 ----
echo ""
echo ">>> 安装 Flash Attention 2..."
# Flash Attention 2 支持 CUDA 12.x / 13.0
pip install flash-attn --no-build-isolation 2>/dev/null && \
    echo "  ✅ Flash Attention 2 安装成功" || \
    echo "  ⚠️  Flash Attention 2 安装失败（将使用 SDPA），不影响训练"

# ---- 4. 验证 ----
echo ""
echo ">>> 验证安装..."
python3 -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA:        {torch.version.cuda}')
print(f'  GPU:         {torch.cuda.get_device_name(0)}')
print(f'  VRAM:        {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')

import transformers
print(f'  Transformers:{transformers.__version__}')

import peft
print(f'  PEFT:        {peft.__version__}')

try:
    import flash_attn
    print(f'  Flash Attn:  {flash_attn.__version__}')
except:
    print(f'  Flash Attn:  NOT INSTALLED (will use SDPA)')

print(f'  All core packages OK ✅')
"

echo ""
echo "============================================"
echo " 环境初始化完成！"
echo ""
echo " 快速测试:  python3 demo_train.py"
echo " 正式训练:  python3 src/train.py"
echo "============================================"
