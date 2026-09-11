#!/bin/bash
# 看门狗: 等checkpoint-2000完整落盘 -> 暂停训练 -> L3全量评测 -> 自动续训
cd /home/dja/桌面/远苍
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

CK=saves/yuancang-full-sft/checkpoint-2000
echo "[$(date '+%H:%M')] 等待 checkpoint-2000 ..."
while [ ! -s "$CK/trainer_state.json" ]; do sleep 60; done
echo "[$(date '+%H:%M')] checkpoint-2000 已落盘, 暂停训练"
systemctl --user stop yc_pipeline
sleep 8

echo "[$(date '+%H:%M')] L3全量评测: GPU0=ckpt-2000, GPU1=原模型基线(同协议零样本)"
python3 eval.py --checkpoint "$CK" --level 3 --batch-size 32 --gpu 0 \
  2>&1 | tee /tmp/eval2000.log &
P0=$!
# 原模型Qwen2.5-Math-7B基线: 同数据同协议(零样本对话式), 用于量化SFT增益
python3 eval.py --checkpoint /home/dja/桌面/远苍/Qwen2.5-Math-7B --level 3 \
  --batch-size 32 --gpu 1 2>&1 | tee /tmp/eval_base.log &
P1=$!
wait $P0 $P1
echo "[$(date '+%H:%M')] 评测结束 (基线+新模型对照完成)"

echo "[$(date '+%H:%M')] 恢复训练 (从checkpoint-2000续训)"
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
echo "[$(date '+%H:%M')] 看门狗任务完成"
