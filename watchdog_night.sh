#!/bin/bash
# 夜间看门狗: 每10分钟巡检, yc_pipeline意外退出则自动续训(最多8次)
cd /home/dja/桌面/远苍
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
unset CUDA_VISIBLE_DEVICES
MAX=8; n=0
while [ $n -lt $MAX ]; do
  sleep 600
  ST=$(systemctl --user is-active yc_pipeline 2>/dev/null)
  if [ "$ST" = "active" ]; then continue; fi
  n=$((n+1))
  echo "[$(date '+%m-%d %H:%M')] 检测到训练退出, 第${n}次自动续训" >> logs/watchdog.log
  systemctl --user reset-failed yc_pipeline 2>/dev/null
  systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh" 2>/dev/null
done
echo "[$(date '+%m-%d %H:%M')] 达到重启上限(8次), 看门狗停止" >> logs/watchdog.log
