#!/bin/bash
cd /home/dja/桌面/远苍
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
unset CUDA_VISIBLE_DEVICES

if [ ! -s train.jsonl ]; then
  > train.jsonl
  while read -r f; do [ -f "$f" ] && cat "$f" >> train.jsonl; done < train_shards.txt
fi

echo "==== 远苍SFT 双卡 $(date) ===="
torchrun --nproc_per_node=2 train.py \
  --data /home/dja/桌面/远苍/train.jsonl \
  --epochs 1 --bsz 4 --accum 4 \
  --lr 1.5e-5 --warmup 1000 \
  --eval-steps 200 --save-steps 200 \
  --output /home/dja/桌面/远苍/saves/yuancang-full-sft
