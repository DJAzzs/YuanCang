#!/bin/bash
# 编排: 等_GPU0的GSM8K出分 -> 杀GPU0评测(防止重复跑AIME) -> 等GPU1的AIME完成 -> 恢复训练
cd /home/dja/桌面/远苍
source venv/bin/activate
while ! grep -q "gsm8k\] acc" /tmp/eval3400.log 2>/dev/null; do sleep 20; done
echo "[$(date '+%H:%M')] GPU0 GSM8K已出分, 终止GPU0评测进程(防重复跑AIME)"
pkill -9 -f "eval.py.*--gpu 0" 2>/dev/null
while ! grep -q "aime2024\] acc\|saved ->" /tmp/eval3400_aime.log 2>/dev/null; do sleep 20; done
echo "[$(date '+%H:%M')] GPU1 AIME2024完成, 恢复训练"
pkill -9 -f "eval.py" 2>/dev/null
sleep 3
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
echo "[$(date '+%H:%M')] 编排完成"
