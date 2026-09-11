#!/bin/bash
# 等待checkpoint-5000完整落盘后自动暂停训练(供5000步对话测试)
cd /home/dja/桌面/远苍
echo "等待 checkpoint-5000 ..."
while [ ! -s saves/yuancang-full-sft/checkpoint-5000/trainer_state.json ]; do
  sleep 60
done
echo "checkpoint-5000 已落盘, 暂停训练 $(date)"
systemctl --user stop yc_pipeline
echo "已暂停. 对话: python3 chat.py --checkpoint saves/yuancang-full-sft/checkpoint-5000 --gpu 0"
echo "恢复: systemd-run --user --unit=yc_pipeline ./run_pipeline_resume.sh"
