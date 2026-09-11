#!/usr/bin/env python3
"""远苍 prepare_data.py (v2, 全流式)

重写目标:
1. 全流式处理: 逐行读取/分批落盘, 不把全部样本攒在内存里 -> 50G+数据不爆RAM
   - 各源用 generator yield dict; 外层按 MAX_MEM_BYTES 控制写出缓冲。
2. 可复现: 所有随机统一走全局 `rng` (seed=42), 不用裸 random.*
3. 独立 eval.jsonl: 从 Dataset/验证集/*.parquet(question,answer) 生成留出评测集,
   与 train 不重叠。
4. --dry / --limit N: 抽样小跑验证字段映射与 schema。

产出:
  <outdir>/train_<shard>.jsonl   (分片, 每个约 MAX_MEM_BYTES)
  <outdir>/eval.jsonl            (留出评测集)
调用方再按需 cat 或由 train.py 直接读分片。
"""
import argparse, glob, json, os, random, sys, time

random.seed(42)          # global RNG -> all sampling reproducible
rng = random.Random(42)
import pyarrow.parquet as pq   # used by generators

# Budget for --dry: each source stops early so schema validation is fast even on huge parquets.
DRY_BUDGET = 50

BASE = "/home/dja/桌面/远苍/Dataset"
SYSTEM_MATH = (
    "你是远苍（YuanCang），一个专注于数学推理与逻辑分析的AI助手。\n"
    "你由独立开发者D.J.A.基于Qwen2.5-Math-7B架构全量微调训练而成。\n\n"
    "你的核心能力：\n"
    "1. 精通算术、代数、几何、微积分、概率统计及组合数学，能够处理竞赛级难题（如AIME难度）。\n"
    "2. 回答时必须展示完整的分步推导（Chain-of-Thought），每一步都要写清依据（如“由余弦定理得”“代入韦达定理”）。\n"
    "3. 善于从多个角度验证答案（如代入检验、特例反推），确保最终结论的严谨性。\n"
    "4. 若题目条件不足或存在歧义，明确指出，绝不凭空猜测。\n\n"
    "交互规则：\n"
    "- 用户输入数学题后，先复述关键条件，再逐步推导，最后用 \\boxed{} 括起最终答案。\n"
    "- 公式请用 LaTeX 渲染（行内用 \\(\\), 行间用 \\[\\]）。\n"
    "- 若需进行复杂数值计算，可借助简短的 Python 伪代码辅助验算，但回答主体仍以数学推导为准。\n"
    "- 如果问题超出你的知识范围，请诚实说明“暂时无法解答”，不编造信息。\n\n"
    "你的使命：用缜密的推理链，帮助用户攻克从基础到奥林匹克的数学难题。"
)
MAX_MEM_BYTES = 512 * 1024 * 1000   # ~500MB per shard buffer (raw json chars)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- generic helpers
def _norm(v, default=""):
    s = str(v or "").strip()
    return s


def _rec(instruction, output, system=SYSTEM_MATH):
    instruction, output = _norm(instruction), _norm(output)
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": "", "output": output,
            "system": system}


# ---------------------------------------------------------------- per-source generators (all yield dict)
def gen_math2():
    path = f"{BASE}/Math2.jsonl"
    with open(path) as f:
        for l in f:
            if not l.strip():
                continue
            try:
                o = json.loads(l); msgs = o.get("messages")
                usr = ass = ""
                if isinstance(msgs, list):
                    for m in msgs:
                        r = str(m.get("role", "")).strip()
                        c = str(m.get("content", "")).strip()
                        if r == "user" and c:
                            usr = c
                        elif r == "assistant" and c:
                            ass = c
                it = _rec(usr, ass)
            except Exception:
                continue
            if it is not None:
                yield it


def gen_math1():
    path = f"{BASE}/Math1.jsonl"
    with open(path) as f:
        for l in f:
            if not l.strip() or rng.random() > 0.3:   # keep ~30%
                continue
            try:
                o = json.loads(l)
                it = _rec(o.get("problem"), str(o.get("formal_statement") or "").replace("\\n", " "))
            except Exception:
                continue
            if it is not None:
                yield it


def _stream_rows_from_table(d):
    """yield (problem, answer) per row from a pandas chunk; respects no global budget here"""
    qcol = "question" if ("question" in d.columns and "problem" not in d.columns) else "problem"
    if qcol not in d.columns:
        return
    acol = None
    for c in ("expected_answer", "answer", "response", "solution"):
        if c in d.columns:
            acol = c; break
    probs, outs = list(d.get(qcol, [])), (None for _ in range(len(d)))
    if acol is not None:
        outs = list(d[acol])
    for p, o_ in zip(probs, outs):
        yield p, ("" if o_ is None else o_)


def gen_nemotron():
    budget = DRY_BUDGET
    for f in sorted(glob.glob(f"{BASE}/Nemotron-SFT-Math-V4/*.parquet")):
        try:
            pf = pq.ParquetFile(f)
            df_names = set(pf.schema.names)
            if "problem" not in df_names:
                continue
            outcol = "response" if "response" in df_names else ("expected_answer" if "expected_answer" in df_names else None)
            for rg_idx in range(pf.metadata.num_row_groups):
                d = pf.read_row_group(rg_idx).to_pandas()
                probs, outs = list(d["problem"]), (None for _ in range(len(d)))
                if outcol is not None:
                    outs = list(d[outcol])
                for p, o_ in zip(probs, outs):
                    it = _rec(p, "" if o_ is None else o_)
                    if it:
                        yield it
                        budget -= 1
                        if budget == 0: return
        except Exception as e:
            log(f"WARN nemotron {f}: {e}")


def gen_openmath():
    budget = DRY_BUDGET
    for f in sorted(glob.glob(f"{BASE}/OpenMathReasoning/*.parquet")):
        try:
            pf = pq.ParquetFile(f)
            cols = set(pf.schema.names)
            if "problem" not in cols:
                continue
            for rg_idx in range(pf.metadata.num_row_groups):
                d = pf.read_row_group(rg_idx).to_pandas()
                if "problem_type" in d.columns:
                    d = d[d["problem_type"] == "has_answer_extracted"]
                probs, ans = list(d.get("problem", [])), list(d.get("expected_answer", []))
                for p, a in zip(probs, ans):
                    it = _rec(p, a)
                    if it:
                        yield it
                        budget -= 1
                        if budget == 0: return
        except Exception as e:
            log(f"WARN openmath {f}: {e}")


def gen_parquet_generic(pattern, outcol="answer"):
    """generic: Math/*.parquet with problem + solution/answer columns"""
    budget = DRY_BUDGET
    for f in sorted(glob.glob(pattern)):
        try:
            pf = pq.ParquetFile(f)
            cols = set(pf.schema.names)
            if "problem" not in cols and "question" not in cols:
                continue
            a_candidates = [c for c in ("solution", "answer", "expected_answer") if c in cols]
            acol = outcol if outcol in cols else (a_candidates[0] if a_candidates else None)
            qcol = "question" if ("question" in cols and "problem" not in cols) else "problem"
            for rg_idx in range(pf.metadata.num_row_groups):
                d = pf.read_row_group(rg_idx).to_pandas()
                probs, outs = list(d.get(qcol, [])), (None for _ in range(len(d)))
                if acol is not None:
                    outs = list(d[acol])
                for p, o_ in zip(probs, outs):
                    it = _rec(p, "" if o_ is None else o_)
                    if it:
                        yield it
                        budget -= 1
                        if budget == 0: return
        except Exception as e:
            log(f"WARN parquet {f}: {e}")


def gen_numinaco():
    budget = DRY_BUDGET
    for f in sorted(glob.glob(f"{BASE}/NuminaMath-CoT/*.parquet")) + \
             sorted(glob.glob(f"{BASE}/Nemotron-PrismMath/*.parquet")):
        try:
            pf = pq.ParquetFile(f)
            cols = set(pf.schema.names)
            prob_col = "problem" if "problem" in cols else ("question" if "question" in cols else None)
            out_cols = [c for c in ("solution", "answer", "response") if c in cols]
            acol = out_cols[0] if out_cols else None
            if prob_col is None:
                continue
            for rg_idx in range(pf.metadata.num_row_groups):
                d = pf.read_row_group(rg_idx).to_pandas()
                probs, outs = list(d.get(prob_col, [])), (None for _ in range(len(d)))
                if acol is not None:
                    outs = list(d[acol])
                for p, o_ in zip(probs, outs):
                    it = _rec(p, "" if o_ is None else o_)
                    if it:
                        yield it
                        budget -= 1
                        if budget == 0: return
        except Exception as e:
            log(f"WARN numina {f}: {e}")


def gen_jsonl_generic(paths, qcol="question", acol="answer"):
    for path in paths:
        try:
            with open(path) as f:
                for l in f:
                    if not l.strip():
                        continue
                    try:
                        o = json.loads(l)
                        it = _rec(o.get(qcol), o.get(acol))
                    except Exception:
                        continue
                    if it is not None:
                        yield it
        except FileNotFoundError as e:
            log(f"WARN {path}: {e}")


def gen_high_medium(pattern, keep_col="accuracy", lt=0.3, rate=1.0):
    """high_part00 / medium: keep rows where accuracy < threshold"""
    budget = DRY_BUDGET
    for f in sorted(glob.glob(pattern)):
        try:
            pf = pq.ParquetFile(f)
            cols = set(pf.schema.names)
            qcol = "problem" if "problem" in cols else ("question" if "question" in cols else None)
            acol = "expected_answer" if "expected_answer" in cols else ("answer" if "answer" in cols else None)
            if not qcol or not acol:
                continue
            for rg_idx in range(pf.metadata.num_row_groups):
                d = pf.read_row_group(rg_idx).to_pandas()
                try:
                    if keep_col and keep_col in cols and lt is not None:
                        col_vals = d[keep_col]
                        mask = []
                        for v in col_vals:
                            try:
                                vv = float(v[-1]) if isinstance(v, (list, tuple)) else float(v)
                            except Exception:
                                vv = 999.0
                            mask.append(vv < lt)
                        import numpy as np
                        d = d[np.array(mask)]
                except Exception as e:
                    log(f"WARN acc filter {f} rg{rg_idx}: {e}")
                probs, ans = list(d.get(qcol, [])), list(d.get(acol, []))
                for p, a in zip(probs, ans):
                    if rng.random() > rate:
                        continue
                    it = _rec(p, a)
                    if it:
                        yield it
                        budget -= 1
                        if budget == 0: return
        except Exception as e:
            log(f"WARN {pattern}: {e}")


# ---------------------------------------------------------------- eval set from 验证集 parquet
_EVAL_QCOLS = ("question", "problem", "instruction")
_EVAL_ACOLS = ("answer", "expected_answer", "response", "solution")


def _iter_eval_rows(d):
    """Yield (q, a) pairs from one row-group DataFrame using flexible column aliases."""
    cols = set(map(str, d.columns))
    qcol = next((c for c in _EVAL_QCOLS if c in cols), None)
    acol = next((c for c in _EVAL_ACOLS if c in cols), None)
    if not qcol or not acol:
        return
    probs, ans = list(d.get(qcol, [])), list(d.get(acol, []))
    for p, a in zip(probs, ans):
        yield p, ("" if a is None else a)


def gen_eval():
    """合并 验证集/ 下所有有效 parquet -> 完整验证集条目。

    未来扩展: 把 GSM8K / MATH-500 / AIME2024 等额外验证集的 parquet/jsonl
    放入 <BASE>/验证集/ (或作为 --eval-sources 追加), 本生成器会自动并入;
    列名兼容 question/problem/instruction + answer/expected_answer/response/solution。
    """
    files = sorted(glob.glob(f"{BASE}/验证集/*.parquet"))
    if not files:
        log("WARN: no parquet found under Dataset/验证集")
    for f in files:
        # skip 0-byte (e.g. medium.parquet) so empty/invalid files never crash merge
        try:
            if os.path.getsize(f) == 0:
                log(f"WARN eval {os.path.basename(f)}: 0 bytes -> skipped")
                continue
        except OSError:
            pass
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:
            log(f"WARN eval {f}: cannot open ({e}) -> skipped")
            continue
        base = os.path.splitext(os.path.basename(f))[0]
        for rg_idx in range(pf.metadata.num_row_groups):
            try:
                d = pf.read_row_group(rg_idx).to_pandas()
            except Exception as e:
                log(f"WARN eval {os.path.basename(f)} rowgroup {rg_idx}: {e}")
                continue
            for p, a in _iter_eval_rows(d):
                it = _rec(p, a)
                if not it:
                    continue
                # tag source so we can audit the composition of the final set
                tagged = dict(it); tagged["source"] = base
                yield tagged


# ---------------------------------------------------------------- streaming sink (bounded memory)
def stream_sink(generator, out_prefix, max_bytes=MAX_MEM_BYTES):
    """Write generator items to sharded jsonl files; each shard <= ~max_bytes raw text.
    Returns total count and list of shard paths. Memory stays bounded."""
    shards = []
    buf = []            # hold serialized lines
    bytes_in_buf = 0
    out_idx = 1
    written = 0

    def flush():
        nonlocal buf, bytes_in_buf, written, out_idx
        if not buf:
            return
        path = f"{out_prefix}_shard{out_idx:04d}.jsonl"
        with open(path, "w") as fp:
            for line in buf:
                fp.write(line + "\n")
        shards.append(path)
        written += len(buf)
        log(f"  flushed {path} ({len(buf):,} items)")
        buf = []
        bytes_in_buf = 0
        out_idx += 1

    for it in generator:
        line = json.dumps(it, ensure_ascii=False)
        sz = len(line.encode("utf-8"))
        if buf and bytes_in_buf + sz > max_bytes:
            flush()
        buf.append(line)
        bytes_in_buf += sz
    flush()
    return written, shards


# ---------------------------------------------------------------- main wiring (source order preserved)
def sources(opt):
    gen = []
    if not opt.exclude.get("math2"):
        log("Gen Math2...");          gen.append((gen_math2(), "Math2"))
    if not opt.exclude.get("math1"):
        log("Gen Math1(30%)...");     gen.append((gen_math1(), "Math1"))
    if not opt.exclude.get("nemotron"):
        log("Gen Nemotron-SFT-Math-V4...");  gen.append((gen_nemotron(), "Nemotron"))
    if not opt.exclude.get("openmath"):
        log("Gen OpenMathReasoning...");     gen.append((gen_openmath(), "OpenMath"))
    if not opt.exclude.get("high_medium"):
        log("Gen high_part00(acc<0.3)...");
        gen.append((gen_high_medium(f"{BASE}/high_part00.parquet", keep_col="accuracy", lt=0.30, rate=1.0), "High"))
        log("Gen medium(acc<0.5 10%)...")
        gen.append((gen_high_medium(f"{BASE}/medium.parquet", keep_col="accuracy", lt=0.50, rate=0.25), "Medium"))
    if not opt.exclude.get("math"):
        log("Gen Math/* generic...");
        for f in sorted(glob.glob(f"{BASE}/Math/*.parquet")):
            gen.append((gen_parquet_generic(f, outcol="answer"), os.path.basename(f)))
        log("Gen math1/* ...")
        for f in sorted(glob.glob(f"{BASE}/math1/*.parquet")):
            gen.append((gen_parquet_generic(f, outcol="solution"), "m1-" + os.path.basename(f)))
    if not opt.exclude.get("numina"):
        log("Gen NuminaMath-CoT/Nemotron-PrismMath...");
        for f in sorted(glob.glob(f"{BASE}/NuminaMath-CoT/*.parquet")) + \
                 sorted(glob.glob(f"{BASE}/Nemotron-PrismMath/*.parquet")):
            gen.append((gen_parquet_generic(f, outcol="solution"), "num-" + os.path.basename(f)))
    if not opt.exclude.get("jsonl"):
        log("Gen MathInstruct...")
        for p in glob.glob(f"{BASE}/MathInstruct*.jsonl"):
            gen.append((gen_jsonl_generic([p], qcol="instruction", acol="output"), os.path.basename(p)))
        log("Gen generic Chat/simpleqa + identity-ish jsonls...")
        paths = []
        for p in glob.glob(f"{BASE}/通用/*.jsonl") + \
                 sorted(glob.glob(f"{BASE}/*.jsonl")):
            if "Math1" in p or "Math2" in p or "identity.json" in p or os.path.basename(p).startswith("train"):
                continue
            paths.append(p)
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            gen.append((gen_jsonl_generic([p], qcol="question", acol="answer"), "gen-" + base))
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/home/dja/桌面/远苍")
    ap.add_argument("--dry", action="store_true",
                    help="only enumerate a few samples from each source to validate schema (no full write)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total items in --dry mode")
    ap.add_argument("--eval-out", default="/home/dja/桌面/远苍/eval.jsonl",
                    help="path for eval leave-out set")
    opt = ap.parse_args()
    os.makedirs(opt.outdir, exist_ok=True)
    opt.exclude = {"openmath": True}   # OpenMathReasoning 只有 expected_answer 无 CoT,默认排除(方案1)

    # ---- dry: sample-validate each source
    if opt.dry:
        log("DRY-RUN (schema validation, seed=42)")
        for g_, name in sources(opt):
            cnt = 0; seen = None
            try:
                for it in g_:
                    cnt += 1
                    if seen is None:
                        seen = it
                    if opt.limit and cnt >= max(2, int(opt.limit)):
                        break
                if seen:
                    log(f"  [{name}] +{cnt} ok; sample keys={list(seen.keys())} "
                        f"instr_len={len(seen['instruction'])} out_len={len(seen['output'])}")
                else:
                    log(f"  [{name}] EMPTY (no valid rows)")
            except Exception as e:
                import traceback
                log(f"  [{name}] ERROR {type(e).__name__}: {e}")
                traceback.print_exc()
        log("DRY-RUN done")
        return

    # ---- full streaming run
    total = 0; all_shards = []
    for i, (g_, name) in enumerate(sources(opt)):
        n, sh = stream_sink(g_, os.path.join(opt.outdir, f"train_part_{i:02d}"), MAX_MEM_BYTES)
        log(f"[{name}] wrote {n:,} items -> {len(sh)} shards")
        total += n
        all_shards.extend(sh)

    # ---- eval set (bounded too; small). Write a single consolidated eval.jsonl
    e, _ = stream_sink(gen_eval(), os.path.join(opt.outdir, "eval_tmp"), MAX_MEM_BYTES)
    log(f"[Eval] wrote {e:,} items")
    # consolidate any shard(s) into one final eval.jsonl (完整正式评测集)
    import glob as _glob
    e_shards = sorted(_glob.glob(os.path.join(opt.outdir, "eval_tmp_shard*.jsonl")))
    if e_shards:
        with open(os.path.join(opt.outdir, "eval.jsonl"), "w") as fp_out:
            for s in e_shards:
                with open(s) as f_in:
                    for line in f_in:
                        fp_out.write(line)
        log(f"[Eval] consolidated -> {os.path.join(opt.outdir,'eval.jsonl')} ({e:,} items)")
    # cleanup tmp eval shards
    for s in e_shards:
        try: os.remove(s)
        except OSError: pass

    # simple manifest
    with open(os.path.join(opt.outdir, "train_shards.txt"), "w") as fp:
        for p in all_shards:
            fp.write(p + "\n")
    log(f"\n=== TOTAL train items: {total:,} across {len(all_shards)} shards ===")
    log("Manifest -> train_shards.txt ; eval tmp file under outdir")


if __name__ == "__main__":
    main()
