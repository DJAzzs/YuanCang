#!/bin/bash
# U谷验证哨兵: checkpoint-4000时暂停, 跑GSM8K+MATH-500, 对比2000步(43.4%), 自动续训
cd /home/dja/桌面/远苍
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CK=saves/yuancang-full-sft/checkpoint-4000
while [ ! -s "$CK/trainer_state.json" ]; do sleep 60; done
echo "[$(date '+%H:%M')] checkpoint-4000落盘, 暂停训练做U谷复测"
systemctl --user stop yc_pipeline; sleep 8
python3 eval.py --checkpoint "$CK" --level 2 --batch-size 32 --gpu 0 > /tmp/eval4000.log 2>&1
echo "[$(date '+%H:%M')] 复测完成:"; grep "acc =" /tmp/eval4000.log
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
echo "[$(date '+%H:%M')] 训练已恢复"
