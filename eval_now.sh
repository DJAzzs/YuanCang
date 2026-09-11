#!/bin/bash
cd /home/dja/桌面/远苍
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CK=/home/dja/桌面/远苍/saves/yuancang-full-sft/checkpoint-2000

echo "[$(date '+%H:%M')] 暂停训练"
systemctl --user stop yc_pipeline
sleep 8

echo "[$(date '+%H:%M')] 双卡评测: GPU0=ckpt-2000 | GPU1=原模型基线"
python3 eval.py --checkpoint "$CK" --level 3 --batch-size 32 --gpu 0 \
  > /tmp/eval2000.log 2>&1 &
P0=$!
python3 eval.py --checkpoint /home/dja/桌面/远苍/Qwen2.5-Math-7B --level 3 \
  --batch-size 32 --gpu 1 > /tmp/eval_base.log 2>&1 &
P1=$!
wait $P0 $P1
echo "[$(date '+%H:%M')] 评测完成"

echo "[$(date '+%H:%M')] 恢复训练(从checkpoint-2000续训)"
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
echo "[$(date '+%H:%M')] 全部完成"
