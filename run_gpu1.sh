#!/bin/bash
cd /home/dja/桌面/远苍
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
unset CUDA_VISIBLE_DEVICES
exec python3 train.py \
  --data /home/dja/桌面/远苍/train.jsonl \
  --epochs 1 --bsz 8 --accum 4 \
  --lr 1.5e-5 --warmup 1000 \
  --eval-steps 200 --save-steps 200 --resume \
  --output /home/dja/桌面/远苍/saves/yuancang-full-sft
