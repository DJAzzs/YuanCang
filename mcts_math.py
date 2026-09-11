#!/usr/bin/env python3
"""远苍 数学推理 MCTS —— 最终跑分用

架构(参考 PHLOX-HotpotQA-Test/src/simulator/mcts.py 的 UCT 风格):
  - State  : 部分解答(推理步列表), 终止条件 = 出现 \\boxed{}
  - Action : 策略模型(远苍)采样生成的"下一个推理步"
  - Value  : PRM(Qwen2.5-Math-PRM-7B)对最新步骤打分, 替代传统rollout
  - 最终答案: 全部终止节点按(访问数×Q)加权投票 + PRM最高分路径

用法:
  python3 mcts_math.py --problem "1+1=?" --iterations 32
  python3 mcts_math.py --problem-file problems.txt --iterations 64 --gpu 0
  python3 mcts_math.py --dry   # 不加载模型, 验证树逻辑
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter

BASE = "/home/dja/桌面/远苍"
POLICY_PATH = f"{BASE}/saves/yuancang-full-sft"          # 训练完成后可指向最佳checkpoint
POLICY_FALLBACK = f"{BASE}/Qwen2.5-Math-7B"
PRM_PATH = f"{BASE}/Qwen2.5-Math-PRM-7B"

STEP_INSTR = ""                                          # 策略自然续写, 无需额外指令
STOP_STRINGS = ["\n\n\n", "\nStep", "\n**Step", "\n## "]
EXTRA0 = "<extra_0>"


# ======================================================================
# 答案工具(与 eval.py 同源逻辑, 独立实现避免导入训练期副作用)
# ======================================================================
def last_boxed(text: str) -> str | None:
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


def extract_answer(text: str) -> str | None:
    return last_boxed(text)


def normalize(s) -> str:
    s = str(s).strip()
    for a, b in [("$", ""), (",", ""), ("\\!", ""), ("\\,", ""), ("\\left", ""),
                 ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("^{\\circ}", ""), ("^\\circ", ""), (" ", ""), ("。", "")]:
        s = s.replace(a, b)
    return s.rstrip(".").lower()


def answers_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    try:
        return abs(float(na) - float(nb)) < 1e-6 * max(1.0, abs(float(nb)))
    except Exception:
        pass
    try:
        import sympy
        def to_sym(x):
            x = normalize(x)
            x = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
            return sympy.sympify(x.replace("\\sqrt", "sqrt").replace("^", "**"),
                                 rational=True)
        return sympy.simplify(to_sym(na) - to_sym(nb)) == 0
    except Exception:
        return False


# ======================================================================
# 状态与节点
# ======================================================================
class StepState:
    """不可变部分解答状态。"""
    __slots__ = ("steps", "terminal", "answer")

    def __init__(self, steps: list[str], terminal: bool = False, answer: str | None = None):
        self.steps = steps
        self.terminal = terminal
        self.answer = answer

    @property
    def text(self) -> str:
        return "\n".join(self.steps)

    def prompt(self, problem: str) -> str:
        return f"user\n{problem}\nassistant\n" + self.text + ("\n" if self.steps else "")


class MCTSNode:
    __slots__ = ("state", "parent", "action", "children", "visits",
                 "total_value", "_value_sq", "depth", "prm_score", "expanded")

    def __init__(self, state: StepState, parent: "MCTSNode | None" = None,
                 action: str = "", depth: int = 0, prm_score: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.total_value = 0.0
        self._value_sq = 0.0
        self.depth = depth
        self.prm_score = prm_score      # 本步自身的PRM分
        self.expanded = False

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits else self.prm_score

    def uct(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.q_value + c * math.sqrt(math.log(max(self.parent.visits, 2)) / self.visits)

    def best_child(self, c: float) -> "MCTSNode":
        return max(self.children, key=lambda ch: ch.uct(c))

    def backup(self, value: float) -> None:
        node: MCTSNode | None = self
        while node is not None:
            node.visits += 1
            node.total_value += value
            node._value_sq += value * value
            node = node.parent


# ======================================================================
# 模型后端: 策略生成 + PRM打分 (批量)
# ======================================================================
class ModelBackend:
    def __init__(self, policy_path: str, prm_path: str, gpu: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.gpu = gpu
        print(f"[backend] 加载策略: {policy_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(policy_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.policy = AutoModelForCausalLM.from_pretrained(
            policy_path, dtype=torch.bfloat16, device_map=f"cuda:{gpu}",
            attn_implementation="sdpa")
        self.policy.eval()

        print(f"[backend] 加载PRM: {prm_path}", flush=True)
        self.prm_tok = AutoTokenizer.from_pretrained(prm_path)
        self.prm = AutoModelForCausalLM.from_pretrained(
            prm_path, dtype=torch.bfloat16, device_map=f"cuda:{gpu}")
        self.prm.eval()
        # <extra_0> 的token id集合(步边界标记)
        self.extra0_ids = set(self.prm_tok.encode(EXTRA0, add_special_tokens=False))
        import torch.nn.functional as F
        self.F = F

    # ---- 策略: 为多个节点各生成K个候选下一步 ----
    def gen_candidates(self, problem: str, nodes: list[MCTSNode],
                       k: int, max_new: int = 224, temperature: float = 0.8):
        """返回 {node_id: [step_text,...]}"""
        torch = self.torch
        prompts, owners = [], []
        for n in nodes:
            for _ in range(k):
                prompts.append(n.state.prompt(problem))
                owners.append(id(n))
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(f"cuda:{self.gpu}")
        with torch.no_grad():
            gen = self.policy.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.95,
                stop_strings=STOP_STRINGS,
                pad_token_id=self.tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)

        out: dict[int, list[str]] = {}
        for own, txt in zip(owners, texts):
            step = txt.strip()
            if not step:
                continue
            out.setdefault(own, []).append(step)
        return out

    # ---- PRM: 批量为一组状态打分(返回最后一步的分) ----
    def prm_score_batch(self, states: list[StepState]) -> list[float]:
        torch, F = self.torch, self.F
        texts, spans = [], []
        for st in states:
            # 每步后接 <extra_0> 作为步边界; 记录每个边界位置
            parts, n_marks = [], 0
            for s in st.steps:
                parts.append(s)
                parts.append("\n" + EXTRA0)
                n_marks += 1
            texts.append("".join(parts) if parts else EXTRA0)
            spans.append(n_marks)
        enc = self.prm_tok(texts, return_tensors="pt", padding=True,
                           return_offsets_mapping=True).to(f"cuda:{self.gpu}")
        with torch.no_grad():
            logits = self.prm(**enc).logits                      # [B, L, 2]
        probs = F.softmax(logits.float(), dim=-1)                # 正确概率 = label 1
        # 对每个 <extra_0> token位置取正确概率, 步分数=该步边界处的均值
        offsets = enc.pop("offset_mapping")
        scores_out = []
        mask_ids = self.extra0_ids
        for b in range(len(states)):
            ids = enc["input_ids"][b]
            marks = [(i, ids[i].item() in mask_ids) for i in range(len(ids))]
            positions = [i for i, is_e0 in marks if is_e0]
            if not positions:
                scores_out.append(0.5)
                continue
            step_scores = []
            for pos in positions:
                p = probs[b, pos]                                 # [2]
                step_scores.append(p[1].item())                   # label1=正确
            scores_out.append(step_scores[-1] if step_scores else 0.5)
        return scores_out


# ======================================================================
# MCTS引擎 (PHLOX风格: UCT + PRM估值替代rollout)
# ======================================================================
class MathMCTSEngine:
    def __init__(self, backend: ModelBackend, problem: str,
                 iterations: int = 32, expansion_k: int = 4,
                 exploration_constant: float = 1.414,
                 max_depth: int = 12, seed: int = 42):
        import random
        self.backend = backend
        self.problem = problem
        self.iterations = iterations
        self.k = expansion_k
        self.c = exploration_constant
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self.terminals: list[MCTSNode] = []      # 全部终止节点(最终投票用)

    def search(self) -> dict:
        root = MCTSNode(StepState([]))
        root.prm_score = 1.0
        root.visits = 1

        for it in range(self.iterations):
            # 1) Selection: UCT下行到可扩展/终止节点
            node = root
            while node.children and not node.state.terminal:
                nxt = node.best_child(self.c)
                if nxt.state.terminal:
                    node = nxt
                    break
                if not nxt.expanded and nxt.depth < self.max_depth:
                    node = nxt
                    break
                node = nxt
            if node.state.terminal:
                # 终止节点重访: 用自身分数backup
                node.backup(node.prm_score)
                continue

            # 2) Expansion: 策略采样K个候选下一步
            cands = self.backend.gen_candidates(self.problem, [node], self.k)[id(node)]
            cands = [c for c in cands if c.strip()][: self.k]
            if not cands:
                node.backup(node.prm_score * 0.5)          # 死路惩罚
                continue

            new_nodes = []
            seen = set()
            for step in cands:
                ans = last_boxed(step)
                terminal = ans is not None
                key = step[:120]
                if key in seen:
                    continue
                seen.add(key)
                child = MCTSNode(
                    StepState(node.state.steps + [step], terminal, ans),
                    parent=node, action=step, depth=node.depth + 1)
                new_nodes.append(child)
            if not new_nodes:
                node.backup(node.prm_score * 0.5)
                continue
            node.children.extend(new_nodes)
            node.expanded = True

            # 3) Evaluation: PRM批量打分(替代rollout)
            scores = self.backend.prm_score_batch([ch.state for ch in new_nodes])
            pick = new_nodes[self.rng.randrange(len(new_nodes))]
            for ch, sc in zip(new_nodes, scores):
                ch.prm_score = sc
                if ch.state.terminal:
                    self.terminals.append(ch)
            # 4) Backpropagation: 用被选中节点(或最优新子)的PRM分
            best_new = max(new_nodes, key=lambda ch: ch.prm_score)
            target = pick if self.rng.random() < 0.5 else best_new
            target.backup(target.prm_score)

        return self._collect_stats(root)

    # ---- 最终答案: 终止节点加权投票 ----
    def final_answer(self) -> tuple[str | None, dict]:
        if not self.terminals:
            return None, {}
        weighted: Counter = Counter()
        for t in self.terminals:
            w = t.visits * max(t.q_value, 0.05) * max(t.prm_score, 0.05)
            if t.state.answer:
                weighted[normalize(t.state.answer)] += w
        if not weighted:
            return None, {}
        best, votes = weighted.most_common(1)[0]
        total_w = sum(weighted.values())
        stats = {"answer": best, "vote_share": round(votes / max(total_w, 1e-9), 3),
                 "n_terminals": len(self.terminals),
                 "distribution": {k: round(v / max(total_w, 1e-9), 3)
                                  for k, v in weighted.most_common(5)}}
        return best, stats

    def _collect_stats(self, root: MCTSNode) -> dict:
        opts = {}
        for ch in sorted(root.children, key=lambda x: -x.q_value)[:8]:
            opts[ch.action[:80]] = {"visits": ch.visits, "q": round(ch.q_value, 3),
                                    "prm": round(ch.prm_score, 3)}
        return {"root_options": opts, "n_terminals": len(self.terminals),
                "iterations": self.iterations}


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None)
    ap.add_argument("--problem-file", default=None)
    ap.add_argument("--iterations", type=int, default=32)
    ap.add_argument("--expansion-k", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--policy", default=POLICY_PATH)
    ap.add_argument("--prm", default=PRM_PATH)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    problems = []
    if args.problem:
        problems.append(args.problem)
    if args.problem_file:
        with open(args.problem_file) as f:
            problems.extend(x.strip() for x in f if x.strip())
    if not problems:
        print("需要 --problem 或 --problem-file")
        sys.exit(1)

    if args.dry:
        # 无模型树逻辑验证: PRM分用随机, 策略候选用固定文本
        import random
        eng = MathMCTSEngine(None, problems[0], iterations=20, seed=42)
        eng.backend = type("B", (), {"gen_candidates": staticmethod(
            lambda p, nodes, k, **kw: {id(n): [f"step {n.depth+1}.{i}" for i in range(k)]
                                       for n in nodes})})()
        import types
        eng.backend.prm_score_batch = types.MethodType(
            lambda self, sts: [min(1.0, 0.2 + 0.15 * len(st.steps)) for st in sts], eng.backend)
        stats = eng.search()
        ans, vs = eng.final_answer()
        print(json.dumps({"dry_answer": ans, "vote": vs, "stats": stats}, ensure_ascii=False, indent=2))
        return

    backend = ModelBackend(args.policy, args.prm, gpu=args.gpu)
    all_results = []
    for prob in problems:
        t0 = time.time()
        eng = MathMCTSEngine(backend, prob, iterations=args.iterations,
                             expansion_k=args.expansion_k)
        stats = eng.search()
        ans, vote = eng.final_answer()
        dt = time.time() - t0
        rec = {"problem": prob[:120], "mcts_answer": ans, "vote": vote,
               "iterations": args.iterations, "seconds": round(dt, 1)}
        all_results.append(rec)
        print(f"\n[题目] {prob[:100]}")
        print(f"[MCTS答案] {ans}   (耗时{dt:.0f}s, 终止节点{vote.get('n_terminals',0)})")
        print(f"[投票分布] {json.dumps(vote.get('distribution', {}), ensure_ascii=False)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
