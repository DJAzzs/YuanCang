#!/bin/bash
cd /home/dja/桌面/远苍
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
CK=saves/yuancang-full-sft/checkpoint-4000
echo "[$(date +%H:%M)] U谷复测: MATH-500全量500条"
python3 eval.py --checkpoint "$CK" --level 3 --tasks math500 --batch-size 32 --gpu 0 --math-subset 500 > /tmp/eval4000.log 2>&1
grep "acc =" /tmp/eval4000.log
echo "[$(date +%H:%M)] 复测完成, 恢复训练"
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline /home/dja/桌面/远苍/run_pipeline_resume.sh
