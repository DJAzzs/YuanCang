#!/bin/bash
# 排队协调器: 等数字季羡林项目的GPU任务退出(显存占用<5GB) -> 启动远苍续训 -> 挂看门狗
cd /home/dja/桌面/远苍
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
echo "[$(date '+%m-%d %H:%M')] 排队中: 等待数字季羡林GPU任务结束..."
while true; do
  BIG=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -oE "[0-9]+ MiB" | grep -oE "[0-9]+" | awk '$1>5000' | wc -l)
  [ "$BIG" -eq 0 ] && break
  sleep 120
done
echo "[$(date '+%m-%d %H:%M')] GPU已释放, 启动远苍续训"
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
sleep 10
systemctl --user reset-failed yc_nightwatch 2>/dev/null
systemd-run --user --unit=yc_nightwatch "$PWD/watchdog_night.sh"
echo "[$(date '+%m-%d %H:%M')] 续训+看门狗已启动"
