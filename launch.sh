#!/bin/bash
# 远苍 - 双卡冒烟测试脚本 (BF16 FSDP, 验证全流程)
#
# 注意: 单卡(world_size=1)下 FSDP full_shard 不会分片参数/优化器状态,
#       7B 全参 BF16+AdamW 需 ~106GB > 95GB 会 OOM。故冒烟也用双卡，
#       与正式训练相同的分片行为，能跑通且更接近真实。
set -euo pipefail

cd /home/dja/桌面/远苍
source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

echo "[launch] 双卡冒烟 (BF16, train_100.jsonl, world_size=2)"

torchrun --nproc_per_node=2 \
  train.py \
    --mode test \
    --data "train_100.jsonl" \
    --gpus 2 \
    2>&1 | tee /tmp/train_smoke.log
