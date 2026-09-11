#!/usr/bin/env python3
"""远苍 评测脚本 (L2/L3)

用法:
  # L2: GSM8K全量 + MATH-500抽100 (单checkpoint)
  python3 eval.py --checkpoint saves/yuancang-full-sft/checkpoint-2000 --level 2

  # L3: 单checkpoint全量 (GSM8K + MATH-500 + AIME2024/2025)
  python3 eval.py --checkpoint saves/yuancang-full-sft/checkpoint-33000 --level 3

  # L3: 遍历目录下全部checkpoint (训后选最佳)
  python3 eval.py --all-checkpoints saves/yuancang-full-sft --level 3

  # 双卡并行: 开两个进程各分一半数据
  python3 eval.py --checkpoint X --level 2 --shard 0 --num-shards 2 --gpu 0
  python3 eval.py --checkpoint X --level 2 --shard 1 --num-shards 2 --gpu 1

  # 冒烟(不加载模型, 验证数据源与答案抽取): python3 eval.py --dry

说明:
  - 训练占满显存期间无法同时跑本脚本(7B需~15GB), L2建议在两checkpoint间隙或训后执行
  - 生成上限: GSM8K 1024 / MATH 2048 / AIME 2048 new tokens (与训练max_len=2048对齐)
"""
import os, re, json, argparse, torch

MODEL_BASE = "/home/dja/桌面/远苍/Qwen2.5-Math-7B"

# ---------------- 数据集 ----------------
def _pick(d, *keys):
    low = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low:
            return str(low[k.lower()])
    return ""

def load_tasks(level, math_subset, only_tasks=None):
    """返回 {task_name: [ {problem, answer} ]}; only_tasks=None时按level取全量"""
    from datasets import load_dataset
    tasks = {}

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"problem": _pick(x, "question"), "answer": _pick(x, "answer").split("####")[-1].strip(),
              "subject": "GSM8K应用题", "level": "1"} for x in ds]
    tasks["gsm8k"] = items

    if level >= 2:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"problem": _pick(x, "problem"), "answer": _pick(x, "answer", "solution"),
                  "subject": _pick(x, "subject", "type") or "MATH综合",
                  "level": _pick(x, "level") or "?"} for x in ds]
        if math_subset and len(items) > math_subset:
            items = items[:math_subset]
        tasks["math500"] = items

    if level >= 3:
        for name, cands in {
            "aime2024": ["AI-MO/aimo-validation-aime", "HuggingFaceH4/aime_2024",
                          "Maxwell-Jia/AIME_2024", "qq8933/AIME_1983_2024"],
            "aime2025": ["math-ai/aime25", "yentinglin/aime_2025", "opencompass/AIME2025"],
        }.items():
            got = False
            for cand in cands:
                ds = None
                try:
                    ds = load_dataset(cand, split="test")
                except Exception:
                    try:
                        dd = load_dataset(cand)
                        ds = dd.get("test") or dd.get("train") or dd.get("validation")
                    except Exception as e:
                        print(f"[debug] {cand}: {str(e)[:60]}")
                        continue
                if ds is None or len(ds) == 0:
                    continue
                cols = ds.column_names
                items = [{"problem": _pick(x, "problem", "question"),
                          "answer": _pick(x, "answer"),
                          "subject": "AIME竞赛", "level": "5"} for x in ds]
                items = [x for x in items if x["problem"]]
                seen, uniq = set(), []
                for x in items:                      # 按题面去重, 防重复计分
                    k = normalize(x["problem"])[:200]
                    if k not in seen:
                        seen.add(k); uniq.append(x)
                items = uniq
                if items:
                    tasks[name] = items
                    print(f"[info] {name} <- {cand} ({len(items)}条去重后, cols={cols})")
                    got = True
                    break
                else:
                    print(f"[debug] {cand}: cols={cols} 但problem列全空")
            if not got:
                print(f"[warn] {name} 数据源均不可用, 跳过")
    if only_tasks:
        tasks = {k: v for k, v in tasks.items() if k in only_tasks}
    return tasks

# ---------------- 答案抽取与比对 ----------------
def last_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth, out = idx + 7, 1, []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch); i += 1
    return "".join(out)

def extract_pred(text):
    b = last_boxed(text)
    if b:
        return b
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return nums[-1].replace(",", "") if nums else None

def normalize(s):
    s = str(s).strip()
    for a, b in [("$",""), ("%",""), (",",""), ("\\!",""), ("\\,",""),
                 ("\\left",""), ("\\right",""), ("\\dfrac","\\frac"), ("\\tfrac","\\frac"),
                 ("^{\\circ}",""), ("^\\circ",""), ("\\ ",""), (" ",""),
                 ("。",""), ("。","")]:
        s = s.replace(a, b)
    return s.rstrip(".").lower()

def to_float(s):
    s = normalize(s)
    s = re.sub(r"\\frac\{(-?[\d.]+)\}\{([\d.]+)\}", r"(\1/\2)", s)
    try:
        return float(s)
    except Exception:
        m = re.fullmatch(r"-?\(?\d+(\.\d+)?(/\d+(\.\d+)?)?\)?", s)
        if m:
            try:
                return eval(s)
            except Exception:
                return None
    return None

def to_sympy(s):
    s = normalize(s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = s.replace("\\sqrt", "sqrt").replace("^", "**")
    s = re.sub(r"\\pi", "pi", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    import sympy
    return sympy.sympify(s, rational=True)

def answers_equal(pred, gt):
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gt)
    if p == g:
        return True
    pf, gf = to_float(p), to_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-6 * max(1.0, abs(gf))
    try:
        import sympy
        pe, ge = to_sympy(p), to_sympy(g)
        if pe is not None and ge is not None:
            return sympy.simplify(pe - ge) == 0
    except Exception:
        pass
    return False

# ---------------- 生成 ----------------
def generate(model, tok, prompts, max_new, batch_size, gpu):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    dev = f"cuda:{gpu}"
    for i in range(0, len(prompts), batch_size):
        chunk = [p for p in prompts[i:i+batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
        done = min(i + batch_size, len(prompts))
        print(f"    {done}/{len(prompts)}", flush=True)
    return outs

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--all-checkpoints", default=None)
    ap.add_argument("--level", type=int, default=2, choices=[2, 3])
    ap.add_argument("--math-subset", type=int, default=100, help="L2时MATH-500抽样数")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=0, help="0=按任务默认")
    ap.add_argument("--quick", type=int, default=0, help="每任务只取前N条(冒烟)")
    ap.add_argument("--tasks", default=None, help="逗号分隔: gsm8k,math500,aime2024,aime2025")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    only = set(args.tasks.split(",")) if args.tasks else None
    tasks = load_tasks(2 if args.level == 2 else 3, args.math_subset if args.level == 2 else 0,
                       only_tasks=only)
    if args.shard or args.num_shards > 1:
        for t in tasks:
            tasks[t] = tasks[t][args.shard::args.num_shards]
    if args.quick:
        for t in tasks:
            tasks[t] = tasks[t][:args.quick]

    if args.dry:
        for t, items in tasks.items():
            print(f"[dry] {t}: {len(items)} 条")
            it = items[0]
            print("   problem:", it["problem"][:60].replace("\n", " "))
            print("   answer :", it["answer"][:60])
            fake = it["problem"][:20] + f" \\boxed{{{it['answer']}}}"
            pred = extract_pred(fake)
            print("   extract:", pred, "| equal:", answers_equal(pred, it["answer"]))
        return

    ckpts = []
    if args.all_checkpoints:
        base = args.all_checkpoints
        ckpts = sorted((os.path.join(base, d) for d in os.listdir(base)
                        if d.startswith("checkpoint-")), key=lambda p: int(p.rsplit("-", 1)[1]))
        print(f"发现 {len(ckpts)} 个checkpoint")
    else:
        ckpts = [args.checkpoint]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs("/home/dja/桌面/远苍/eval_results", exist_ok=True)

    for ck in ckpts:
        print(f"\n===== 评测 {ck} =====", flush=True)
        tok = AutoTokenizer.from_pretrained(ck if os.path.exists(os.path.join(ck, "tokenizer_config.json")) else MODEL_BASE)
        model = AutoModelForCausalLM.from_pretrained(
            ck, dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}",
            attn_implementation="sdpa")
        model.eval()

        result = {"checkpoint": ck, "level": args.level}
        for tname, items in tasks.items():
            max_new = args.max_new or {"gsm8k": 1024, "math500": 2048}.get(tname, 2048)
            prompts = [f"user\n{x['problem']}\nassistant\n" for x in items]
            print(f"[{tname}] {len(items)} 条, max_new={max_new}", flush=True)
            outs = generate(model, tok, prompts, max_new, args.batch_size, args.gpu)
            detail, correct = [], 0
            for o, x in zip(outs, items):
                pred = extract_pred(o)
                ok = answers_equal(pred, x["answer"])
                correct += ok
                detail.append({"idx": x.get("idx", len(detail)), "subject": x.get("subject", ""),
                               "level": str(x.get("level", "?")), "correct": bool(ok),
                               "pred": pred, "gt": x["answer"]})
            acc = correct / max(1, len(items))
            result[tname] = {"acc": round(acc * 100, 2), "correct": correct,
                             "total": len(items), "items": detail}
            print(f"[{tname}] acc = {acc*100:.2f}% ({correct}/{len(items)})", flush=True)

        tag = os.path.basename(ck.rstrip("/")) or "ckpt"
        out_json = f"/home/dja/桌面/远苍/eval_results/{tag}_L{args.level}_s{args.shard}.json"
        with open(out_json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("saved ->", out_json)
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
