#!/bin/bash
cd /home/dja/桌面/远苍
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CK=/home/dja/桌面/远苍/saves/yuancang-full-sft/checkpoint-11400
echo "[$(date '+%H:%M')] 暂停训练"
systemctl --user stop yc_pipeline
sleep 8
echo "[$(date '+%H:%M')] 双卡分片评测: GSM8K(1319)+MATH-500(500)"
python3 eval.py --checkpoint "$CK" --level 2 --tasks gsm8k,math500 --batch-size 32 --gpu 0 --shard 0 --num-shards 2 > /tmp/eval11400_s0.log 2>&1 &
P0=$!
python3 eval.py --checkpoint "$CK" --level 2 --tasks gsm8k,math500 --batch-size 32 --gpu 1 --shard 1 --num-shards 2 > /tmp/eval11400_s1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "[$(date '+%H:%M')] 评测完成:"
grep -h "acc =" /tmp/eval11400_s0.log /tmp/eval11400_s1.log
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
echo "[$(date '+%H:%M')] 训练已恢复"
