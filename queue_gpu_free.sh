#!/bin/bash
# GPU空闲监听: 连续3分钟无大任务(>5GB)才启动续训+看门狗; 否则继续排队
cd /home/dja/桌面/远苍
LOG=logs/queue_gpu.log
echo "[$(date '+%m-%d %H:%M')] 监听启动: 等待GPU空闲(宽限3分钟确认)..." >> $LOG
while true; do
  BIG=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -oE '[0-9]+ MiB' | grep -oE '[0-9]+' | awk '$1>5000' | wc -l)
  if [ "$BIG" -eq 0 ]; then
    echo "[$(date '+%m-%d %H:%M')] GPU空闲, 宽限3分钟确认无新任务..." >> $LOG
    sleep 180
    BIG2=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -oE '[0-9]+ MiB' | grep -oE '[0-9]+' | awk '$1>5000' | wc -l)
    if [ "$BIG2" -eq 0 ]; then
      echo "[$(date '+%m-%d %H:%M')] 确认空闲, 启动续训+看门狗" >> $LOG
      break
    fi
    echo "[$(date '+%m-%d %H:%M')] 宽限期检测到新任务, 继续排队..." >> $LOG
  fi
  sleep 120
done
systemctl --user reset-failed yc_pipeline 2>/dev/null
systemd-run --user --unit=yc_pipeline "$PWD/run_pipeline_resume.sh"
sleep 10
systemctl --user reset-failed yc_nightwatch 2>/dev/null
systemd-run --user --unit=yc_nightwatch "$PWD/watchdog_night.sh"
echo "[$(date '+%m-%d %H:%M')] 续训+看门狗已启动" >> $LOG
