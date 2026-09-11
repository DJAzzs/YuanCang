#!/bin/bash
# 远苍训练实时监控: ./monitor.sh [once] [刷新秒数,默认5]
BASE=/home/dja/桌面/远苍
CKPT_DIR=$BASE/saves/yuancang-full-sft
ONCE=${1:-}
REFRESH=${2:-5}
TOTAL=$(( $(wc -l < $BASE/train.jsonl 2>/dev/null || echo 3200000) / 32 ))
JTMP=$(mktemp /tmp/yc_live.XXXX)

snapshot() {
  clear
  echo "╔═══════════ 远苍 SFT 实时监控 $(date '+%m-%d %H:%M:%S') ═══════════╗"

  # 一次性解码journald blob -> 实时CUR+MET
  journalctl --user -u yc_pipeline --since "30 min ago" -o json 2>/dev/null \
    | python3 "$BASE/live_status.py" > "$JTMP" 2>/dev/null

  echo "╟─ LIVE (实时流) ───────────────────────────────"
  grep "^CUR|" "$JTMP" | sed 's/^CUR|/║  /' || true
  grep -q "^CUR|" "$JTMP" || echo "║  (30分钟内无进度输出)"

  LATEST=$(ls -d $CKPT_DIR/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  if [ -n "$LATEST" ]; then
    PCT=$(( LATEST * 100 / (TOTAL > 0 ? TOTAL : 1) ))
    echo "║ 存档: checkpoint-$LATEST / $TOTAL (${PCT}%)"
  fi

  echo "╟─ 训练指标 (最近5条, 每5步更新) ───────────────"
  grep "^MET|" "$JTMP" | tail -5 | sed 's/^MET|/║  /'
  [ "$(grep -c '^MET|' "$JTMP")" = "0" ] && echo "║  (暂无)"

  echo "╟─ L1 验证集 (每200步, 存档时更新) ─────────────"
  if [ -n "$LATEST" ]; then
    python3 - "$CKPT_DIR/checkpoint-$LATEST/trainer_state.json" <<'PY'
import json, sys, os, time, math
path = sys.argv[1]
for _ in range(3):                      # 存档写入中(trainer_state未落盘)则短暂等待重试
    if os.path.exists(path) and os.path.getsize(path) > 2:
        break
    time.sleep(1)
else:
    print("║  checkpoint-" + path.rsplit("-",1)[1].split("/")[0] + " 写入中...")
    sys.exit(0)
try:
    st = json.load(open(path))
    ev = [h for h in st.get("log_history", []) if "eval_loss" in h]
    if ev:
        for h in ev[-3:]:
            ppl = math.exp(h.get("eval_loss", 0))
            print(f"║  step {h.get('step',0):>6} | eval_loss {h.get('eval_loss',0):.4f} | PPL {ppl:.2f}")
    else:
        print("║  首次评测在step 200")
except Exception as e:
    print("║  trainer_state读取失败:", e)
PY
  fi

  echo "╟─ GPU 状态 ────────────────────────────────────"
  nvidia-smi --query-gpu=index,power.draw,power.limit,clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "║  GPU%s: %s / %sW | %s MHz | %s°C | %s%%\n", $1,$2,$3,$4,$5,$6}'
  for i in 0 1; do
    nvidia-smi -q -i $i -d PERFORMANCE 2>/dev/null | grep -m1 "SW Power Cap" | grep -q "Active" && \
      echo "║  ⚠ GPU$i 撞功耗墙(频率受限)" || true
  done

  echo "╟─ Checkpoints ─────────────────────────────────"
  ls -d $CKPT_DIR/checkpoint-* 2>/dev/null | sed 's/.*\//║  /'
  [ -f "$BASE/eval.jsonl" ] && echo "║  L1数据: $(wc -l < $BASE/eval.jsonl) 条(验证集)"
  echo "╚═══════════════════════════════════════════════╝"
  systemctl --user is-active yc_pipeline | xargs -I{} echo "服务状态: {}"
}

if [ "$ONCE" = "once" ]; then
  snapshot
  rm -f "$JTMP"
else
  trap 'rm -f "$JTMP"; exit' INT TERM
  while true; do snapshot; sleep "$REFRESH"; done
fi
