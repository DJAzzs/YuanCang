import sys, json, re
step = loss = speed = None
pairs = []          # (step, loss, grad, lr)
last_step = None
PAT_LOSS = re.compile(r"\{'loss': '([\d.]+)', 'grad_norm': '([\d.]+)', 'learning_rate': '([^']+)'")
for l in sys.stdin:
    try:
        d = json.loads(l)
        m = d.get("MESSAGE")
        if isinstance(m, list):
            m = bytes(m).decode("utf-8", "replace")
        m = str(m).replace("\r", "\n")
        for line in m.split("\n"):
            s = re.search(r"(\d+)/33262 \[([^]]*)\]", line)
            if s:
                last_step = int(s.group(1))
                sp = re.search(r"([\d.]+)s/it", s.group(2))
                if sp: speed = sp.group(1)
            l2 = PAT_LOSS.search(line)
            if l2:
                pairs.append((last_step,) + l2.groups())
                loss = l2.group(1)
    except Exception:
        pass
step = last_step
if step:
    try:
        eta_h = (33262 - step) * float(speed) / 3600 if speed else 0
        print(f"CUR|step {step}/33262 | loss {loss or '-'} | {speed or '-'}s/it | ETA {eta_h:.1f}h")
    except Exception:
        print(f"CUR|step {step}/33262 | loss {loss or '-'}")
else:
    print("CUR|(10分钟内无进度输出)")
for p in pairs[-5:]:
    st, lo, gr, lr = p
    lr_f = float(lr) if lr else 0
    print(f"MET|step {st:>6} | loss {float(lo):.4f} | grad {float(gr):.2f} | lr {lr_f:.2e} | epoch {st/33262:.3f}")
