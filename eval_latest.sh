#!/bin/bash
# 评测最新checkpoint: 暂停训练 -> 复制到安全路径(防滚动删除) -> 双卡分片评测GSM8K+MATH-500 -> 恢复训练
cd /home/dja/桌面/远苍
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
BASE=saves/yuancang-full-sft
LATEST=$(ls -d $BASE/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
echo "[$(date '+%H:%M')] 最新checkpoint: $LATEST"
SAFE=/home/dja/桌面/远苍/eval_ckpt_snapshot
rm -rf "$SAFE"; cp -r "$BASE/checkpoint-$LATEST" "$SAFE"
echo "[$(date '+%H:%M')] 已复制到 $SAFE (防滚动删除)"
systemctl --user stop yc_pipeline
sleep 8
echo "[$(date '+%H:%M')] 双卡分片评测: GSM8K+MATH-500"
python3 eval.py --checkpoint "$SAFE" --level 2 --tasks gsm8k,math500 --batch-size 32 --gpu 0 --shard 0 --num-shards 2 > /tmp/eval_latest_s0.log 2>&1 &
P0=$!
python3 eval.py --checkpoint "$SAFE" --level 2 --tasks gsm8k,math500 --batch-size 32 --gpu 1 --shard 1 --num-shards 2 > /tmp/eval_latest_s1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "[$(date '+%H:%M')] 评测完成:"
grep -h "acc =" /tmp/eval_latest_s0.log /tmp/eval_latest_s1.log
echo "[$(date '+%H:%M')] 恢复训练"
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
echo "[$(date '+%H:%M')] 完成"
