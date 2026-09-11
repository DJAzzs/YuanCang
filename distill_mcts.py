#!/usr/bin/env python3
"""MCTS轨迹蒸馏: 题库逐题搜索 -> 验证通过的树轨迹 -> 二次微调训练集

数据源: 桌面dapo parquet(prompt列表+reward_model.ground_truth)
流程: 每题MCTS搜索 -> 终止节点(按Q×visits×PRM排序)取Top轨迹
      -> 答案对ground_truth硬验证 + 路径PRM最低分软过滤 -> 写SFT jsonl
用法:
  python3 distill_mcts.py --limit 100 --iterations 32            # 冒烟/正式
  python3 distill_mcts.py --dry                                  # 无模型验证
"""
import os, re, json, time, argparse, random
import pandas as pd

BASE = "/home/dja/桌面/远苍"
DAPO = "/home/dja/桌面/dapo-math-17k.parquet"
OUT_DEFAULT = f"{BASE}/distill_mcts.jsonl"

SYSTEM_MATH = (
    "你是远苍（YuanCang），一个专注于数学推理与逻辑分析的AI助手。\n"
    "你由独立开发者D.J.A.基于Qwen2.5-Math-7B架构全量微调训练而成。\n\n"
    "你的核心能力：\n"
    "1. 精通算术、代数、几何、微积分、概率统计及组合数学，能够处理竞赛级难题（如AIME难度）。\n"
    "2. 回答时必须展示完整的分步推导（Chain-of-Thought），每一步都要写清依据。\n"
    "3. 善于从多个角度验证答案，确保最终结论的严谨性。\n"
    "4. 若题目条件不足或存在歧义，明确指出，绝不凭空猜测。\n\n"
    "交互规则：\n"
    "- 用户输入数学题后，先复述关键条件，再逐步推导，最后用 \\boxed{} 括起最终答案。\n"
    "- 公式请用 LaTeX 渲染。\n"
    "- 如果问题超出你的知识范围，请诚实说明，不编造信息。\n\n"
    "你的使命：用缜密的推理链，帮助用户攻克从基础到奥林匹克的数学难题。"
)

# ---- 从dapo parquet提取题库 {problem, gt} ----
def load_dapo(limit, seed=42):
    import numpy as np
    df = pd.read_parquet(DAPO)
    # 1.79M行iterrows太慢: 先随机采样20万行再遍历, 唯一题早退
    if len(df) > 200000:
        df = df.sample(n=200000, random_state=seed)
    items, seen = [], set()
    for _, r in df.iterrows():
        if limit and len(items) >= max(limit * 5, 2000):
            break
        p = r.get("prompt")
        prob = ""
        if isinstance(p, (list, pd.Series, np.ndarray)):
            for m in p:
                if isinstance(m, dict) and m.get("role") == "user":
                    prob = str(m.get("content", ""))
                    break
        elif isinstance(p, str):
            prob = p
        rm = r.get("reward_model")
        gt = str(rm.get("ground_truth", "")).strip() if isinstance(rm, dict) else ""
        prob = prob.strip()
        if not prob or not gt:
            continue
        # DAPO官方包装清洗: 提取纯题面
        m = re.search(r"answer to the problem\.\n\n(.+?)\n\nRemember to", prob, re.S)
        if m:
            prob = m.group(1).strip()
        key = prob[:200].lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({"problem": prob, "gt": gt})
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items

# ---- 从MCTS树提取Top轨迹(按验证/Q/PRM排序, 含路径PRM最低分) ----
def extract_trajectories(eng, gt, top_n=2, prm_min=0.25):
    import mcts_math as M
    out = []
    # 终止节点按(投票答案正确 × Q × visits × prm)打分
    scored = []
    for t in eng.terminals:
        if not t.state.answer:
            continue
        correct = M.answers_equal(t.state.answer, gt)
        score = (2.0 if correct else 0.0) * max(t.q_value, 0.05) * \
                (t.visits + 1) * max(t.prm_score, 0.05)
        scored.append((score, t, correct))
    scored.sort(key=lambda x: -x[0])
    seen, picked = set(), []
    for score, t, correct in scored:
        key = M.normalize(t.state.answer)[:120]
        if key in seen:
            continue
        seen.add(key)
        # 路径PRM最低分(软过滤)
        path_min = t.prm_score
        node = t
        while node.parent is not None:
            node = node.parent
            if node.depth > 0:
                path_min = min(path_min, node.prm_score)
        picked.append((t, score, correct, path_min))
        if len(picked) >= top_n:
            break
    for t, score, correct, path_min in picked:
        if not correct:
            continue                    # 硬过滤: 答案必须匹配ground_truth
        if path_min < prm_min:
            continue                    # 软过滤: 路径上不能有太烂的步
        out.append({"steps": t.state.steps, "answer": t.state.answer,
                    "prm_min": round(path_min, 3), "score": round(score, 2)})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="题库抽题数")
    ap.add_argument("--iterations", type=int, default=32)
    ap.add_argument("--expansion-k", type=int, default=4)
    ap.add_argument("--top-n", type=int, default=2, help="每题最多保留轨迹数")
    ap.add_argument("--prm-min", type=float, default=0.25)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--policy", default=f"{BASE}/saves/yuancang-full-sft")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    bank = load_dapo(args.limit)
    print(f"[bank] 有效去重题 {len(bank)} 道", flush=True)
    if args.dry:
        import mcts_math as M
        eng = M.MathMCTSEngine(None, bank[0]["problem"], iterations=20, seed=42)
        from mcts_math import StepState as _SS, MCTSNode as _MN
        eng.root = _MN(_SS([]))
        eng.backend = type("B", (), {"gen_candidates": staticmethod(
            lambda p, nodes, k, **kw: {id(n): [f"推导第{n.depth+1}步.{i}" for i in range(k)]
                                       for n in nodes})})()
        import types
        eng.backend.prm_score_batch = types.MethodType(
            lambda self, sts: [min(1.0, 0.2 + 0.15*len(st.steps)) for st in sts], eng.backend)
        gt = bank[0]["gt"] if bank else "1/2"
        # dry模式下无\boxed → 强造一个终止节点验证抽取逻辑
        from mcts_math import StepState, MCTSNode
        t = MCTSNode(StepState(eng.root.state.steps + ["所以答案是 \\boxed{" + gt + "}"],
                                True, gt), prm_score=0.9)
        eng.terminals.append(t)
        trajs = extract_trajectories(eng, gt, top_n=2, prm_min=0.0)
        print(f"[dry] 抽取轨迹 {len(trajs)} 条:", json.dumps(trajs, ensure_ascii=False)[:300])
        return

    import mcts_math as M
    backend = M.ModelBackend(args.policy, args.prm or f"{BASE}/Qwen2.5-Math-PRM-7B", gpu=args.gpu)

    out_path = args.out
    n_prob = n_traj = 0
    t0 = time.time()
    with open(out_path, "a") as fout:
        for i, item in enumerate(bank):
            try:
                eng = M.MathMCTSEngine(backend, item["problem"],
                                       iterations=args.iterations,
                                       expansion_k=args.expansion_k)
                eng.search()
                trajs = extract_trajectories(eng, item["gt"],
                                             top_n=args.top_n, prm_min=args.prm_min)
            except Exception as e:
                print(f"[{i+1}] ERR {str(e)[:60]}", flush=True)
                continue
            for ti, tr in enumerate(trajs):
                fout.write(json.dumps({
                    "instruction": item["problem"],
                    "input": "",
                    "output": "\n".join(tr["steps"]),
                    "system": SYSTEM_MATH,
                    "meta": {"prm_min": tr["prm_min"], "score": tr["score"], "traj": ti},
                }, ensure_ascii=False) + "\n")
                n_traj += 1
            n_prob += 1
            fout.flush()
            el = time.time() - t0
            print(f"[{i+1}/{len(bank)}] 轨迹+{len(trajs)} (累计{n_traj}) "
                  f"| {el/ (i+1):.0f}s/题 | ETA {(el/(i+1))*(len(bank)-i-1)/3600:.1f}h", flush=True)

    print(f"[完成] {n_prob}题 → {n_traj}条MCTS轨迹 → {out_path}")

if __name__ == "__main__":
    main()
