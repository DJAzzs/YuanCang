#!/bin/bash
# 远苍 - NVFP4 QAT 正式训练启动脚本 (双卡)
# 用法: bash launch_full_sft.sh [--quant bf16]
set -euo pipefail

cd /home/dja/桌面/远苍
source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
# FSDP 通信重叠 (依优化表)
export CUDA_LAUNCH_BLOCKING=0

QUANT="nvfp4"
if [[ "${1:-}" == "--quant" && -n "${2:-}" ]]; then
    QUANT="$2"
fi

echo "[launch] quant=${QUANT} nproc=2 (global batch 32, seq len 8192)"

torchrun --nproc_per_node=2 \
  train_nvfp4.py \
    --mode full \
    --data "train_part_*.jsonl" \
    --gpus 2 \
    --quant "${QUANT}" \
    2>&1 | tee /tmp/train_full.log
