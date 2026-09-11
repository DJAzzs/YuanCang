#!/bin/bash
cd /home/dja/桌面/远苍
source venv/bin/activate
torchrun --nproc_per_node=2 train.py --data train_100.jsonl --max-steps 3 --bsz 2 --accum 1 --optim adamw_torch --output saves/test_final
