#!/bin/bash
# 10%检查点(step 3400): 仅GSM8K + AIME2024, 评完自动续训
cd /home/dja/桌面/远苍
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CK=saves/yuancang-full-sft/checkpoint-3400
while [ ! -s "$CK/trainer_state.json" ]; do sleep 60; done
echo "[$(date '+%H:%M')] checkpoint-3400落盘, 暂停训练"
systemctl --user stop yc_pipeline
sleep 8
python3 eval.py --checkpoint "$CK" --level 3 --tasks gsm8k,aime2024 \
  --batch-size 32 --gpu 0 > /tmp/eval3400.log 2>&1
echo "[$(date '+%H:%M')] 评测完成:"; grep "acc =" /tmp/eval3400.log
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
echo "[$(date '+%H:%M')] 训练已恢复"
