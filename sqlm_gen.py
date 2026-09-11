#!/usr/bin/env python3
"""SQLM落地: 自提问数据引擎 (Self-Questioning Language Models, arXiv 2508.03682)

务实版实现(拒绝采样近似RL, 工程量小一个量级):
  提问者: 远苍按主题+难度提示生成新题 (采样)
  解题者: 远苍对每题G次采样求解
  难度门控: pass_rate ∈ [0.125, 0.875] 保留 (SQLM的proposer奖励的采样近似:
            太简单=0.875+, 太难=0.125- 都无学习价值)
  解答目标: 多数投票答案中PRM分最高的一致解 (solver奖励的过滤近似)

输出: SFT jsonl (可并入二次微调语料, 与MCTS轨迹/RFT样本混配)
用法:
  python3 sqlm_gen.py --dry
  python3 sqlm_gen.py --rounds 3 --per-topic 12 --gpu 0
"""
import os, re, json, time, argparse, random
from collections import Counter

BASE = "/home/dja/桌面/远苍"

TOPICS = [
    ("代数方程与不等式", "中考-高考难度"),
    ("数列与递推", "高考-竞赛一试难度"),
    ("三角函数与解三角形", "高考难度"),
    ("立体几何与空间向量", "高考难度"),
    ("解析几何(圆锥曲线)", "高考压轴难度"),
    ("排列组合与二项式定理", "竞赛一试难度"),
    ("概率与期望", "竞赛一试难度"),
    ("初等数论(整除/同余/不定方程)", "竞赛难度"),
    ("函数与导数综合", "高考压轴难度"),
    ("复数与多项式", "竞赛难度"),
    ("组合数学(计数/图论/博弈)", "竞赛二试/AIME难度"),
    ("整数规划与丢番图方程", "AIME难度"),
    ("平面几何(圆/相似/共点共线)", "联赛二试难度"),
    ("数论竞赛题(费马小定理/中国剩余)", "AIME难度"),
    ("级数求和与极限估算", "大学低年级难度"),
]

PROPOSER_PROMPT = """你是一位顶级数学竞赛命题人。请围绕主题「{topic}」命制 {n} 道全新的数学题。

要求:
1. 每道题必须完整、自洽, 有唯一确定的数值或简洁解析答案
2. 难度定位: {difficulty}
3. 题目之间考点/情境/数字不得重复
4. 只输出题目, 每题以 <Q> 开头, 不要给出解答

开始命题:"""

SOLVE_PROMPT = "user\n{q}\nassistant\n"


def extract_questions(text):
    qs = []
    for chunk in re.split(r"<Q>", text)[1:]:
        q = chunk.strip()
        q = re.sub(r"^\d+[.、)]\s*", "", q)
        if 30 < len(q) < 1500 and ("?" in q or "？" in q or "求" in q or "计算" in q
                                    or "证明" in q or "多少" in q or "值" in q):
            qs.append(q)
    return qs


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


def normalize(s):
    s = str(s).strip()
    for a, b in [("$", ""), (",", ""), (" ", ""), ("\\left", ""), ("\\right", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("^\\circ", ""),
                 ("^{\\circ}", ""), ("。", "")]:
        s = s.replace(a, b)
    return s.rstrip(".").lower()


def answers_equal(a, b):
    if a is None or b is None:
        return False
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    try:
        return abs(float(na) - float(nb)) < 1e-6 * max(1.0, abs(float(nb)))
    except Exception:
        return False


class Engine:
    """单模型双角色: 同一远苍分别当提问者与解题者"""

    def __init__(self, policy_path, gpu=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.gpu = gpu
        print(f"[sqlm] 加载 {policy_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(policy_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            policy_path, dtype=torch.bfloat16,
            device_map=f"cuda:{gpu}", attn_implementation="sdpa")
        self.model.eval()

    def _gen(self, prompts, max_new, temperature):
        torch = self.torch
        self.tok.padding_side = "left"
        outs = []
        for i in range(0, len(prompts), 32):
            enc = self.tok(prompts[i:i+32], return_tensors="pt",
                           padding=True, truncation=True, max_length=3072).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new, do_sample=True,
                    temperature=temperature, top_p=0.95,
                    pad_token_id=self.tok.eos_token_id)
            outs.extend(self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True))
        return outs

    def propose(self, topic, n, temperature=1.0):
        prompt = PROPOSER_PROMPT.format(topic=topic, n=n, difficulty="适中偏难, 区分度高")
        outs = self._gen([prompt], max_new=1600, temperature=temperature)
        return extract_questions(outs[0])

    def solve(self, questions, g=8, temperature=0.8, max_new=1536):
        prompts = [SOLVE_PROMPT.format(q=q) for q in questions for _ in range(g)]
        raws = self._gen(prompts, max_new=max_new, temperature=temperature)
        per_q = [raws[i*g:(i+1)*g] for i in range(len(questions))]
        results = []
        for q, sols in zip(questions, per_q):
            answers = [last_boxed(s) for s in sols]
            answers = [a for a in answers if a]
            if not answers:
                results.append(None); continue
            cnt = Counter(normalize(a) for a in answers)
            maj, votes = cnt.most_common(1)[0]
            pass_rate = votes / len(sols)
            # 多数派一致解中取最长者作为SFT目标(通常最完整)
            maj_sols = [s for s, a in zip(sols, answers) if a and normalize(a) == maj]
            target = max(maj_sols, key=len) if maj_sols else None
            results.append({"question": q, "majority": maj, "pass_rate": pass_rate,
                            "target": target, "g": len(sols)})
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{BASE}/saves/yuancang-full-sft")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3, help="每主题轮数(不同采样批次)")
    ap.add_argument("--per-topic", type=int, default=12, help="每轮每主题出题数")
    ap.add_argument("--g", type=int, default=8, help="每题解题采样数")
    ap.add_argument("--out", default=f"{BASE}/sqlm_data.jsonl")
    ap.add_argument("--weights", default=None, help="失分分析权重JSON(weakness_report.py产出)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.dry:
        print("[dry] 主题数:", len(TOPICS), "| 轮数:", args.rounds,
              "| 每轮出题:", args.per_topic, "| 采样G:", args.g)
        fake_q = "已知 $x+y=3, xy=2$, 求 $x^2+y^2$ 的值。"
        import types
        eng = Engine.__new__(Engine)
        eng._gen = types.MethodType(lambda self, ps, max_new, temperature=0.7: [
            "<Q>" + fake_q] if "命题" in ps[0] else
            [f"由韦达定理...\\boxed{{5}}" if i % 4 else f"\\boxed{{5}} 工业{i}"
             for i in range(len(ps))], eng)
        qs = eng.propose("代数方程", 3)
        print("[dry] propose输出题目数:", len(qs))
        res = eng.solve([fake_q] * 2, g=8)
        kept = [r for r in res if r and 0.125 <= r["pass_rate"] <= 0.875]
        print("[dry] 难度门控保留:", len(kept), "/", len(res), "(预期: 多数投票全一致=太简单被剔除)")
        return

    topic_w = {t: 1.0 for t, _ in TOPICS}
    if args.weights and os.path.exists(args.weights):
        loaded = json.load(open(args.weights))
        for k, v in loaded.items():
            if k in topic_w:
                topic_w[k] = v
        print(f"[weights] 失分加权: {topic_w}", flush=True)

    eng = Engine(args.checkpoint, gpu=args.gpu)
    out, seen = [], set()
    t0 = time.time()
    for rnd in range(args.rounds):
        for topic, diff in TOPICS:
            n_q = max(2, round(args.per_topic * topic_w.get(topic, 1.0)))
            qs = eng.propose(topic, n_q)
            qs = [q for q in qs if q[:120].lower() not in seen]
            if not qs:
                continue
            for q in qs:
                seen.add(q[:120].lower())
            res = eng.solve(qs, g=args.g)
            kept = 0
            for r in res:
                if r is None:
                    continue
                if not (0.125 <= r["pass_rate"] <= 0.875):
                    continue                     # SQLM难度门控
                out.append({"instruction": r["question"], "input": "",
                            "output": r["target"],
                            "system": f"meta:pass_rate={r['pass_rate']:.2f}",
                            "topic": topic})
                kept += 1
            el = time.time() - t0
            print(f"[{rnd+1}/{args.rounds}] {topic}(×{topic_w.get(topic,1.0):.1f}): 出题{len(qs)} 保留{kept} "
                  f"| 累计{len(out)} | {el/60:.0f}min", flush=True)
            with open(args.out, "w") as f:
                for it in out:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[完成] SQLM自提问数据 {len(out)} 条 → {args.out}")

if __name__ == "__main__":
    main()
