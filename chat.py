#!/usr/bin/env python3
"""远苍 对话测试: python3 chat.py --checkpoint saves/yuancang-full-sft/checkpoint-5000 [--gpu 0]"""
import torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=0.9)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.checkpoint)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.checkpoint, dtype=torch.bfloat16,
    device_map=f"cuda:{args.gpu}", attn_implementation="sdpa")
model.eval()
print(f"已加载 {args.checkpoint} | 输入quit退出 | 多轮对话(保留最近3轮上下文)")

history = []
while True:
    try:
        q = input("\n你> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q or q.lower() in ("quit", "exit", "q"):
        break
    ctx = "".join(f"user\n{u}\nassistant\n{a}\n" for u, a in history[-3:])
    prompt = f"{ctx}user\n{q}\nassistant\n"
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=args.max_new,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"远苍> {resp}")
    history.append((q, resp))
