#!/usr/bin/env python3
"""失分分析: 从eval_results提取逐题明细 -> 考点错误率排名 -> SQLM主题权重

用法: python3 weakness_report.py eval_results/checkpoint-XXXX_L3_s0.json
输出: 终端报告 + weights_for_sqlm.json (供 sqlm_gen.py --weights 使用)
"""
import json, sys, argparse

# MATH-500 subject → sqlm_gen.py TOPICS 的映射
SUBJECT2TOPIC = {
    "algebra": ["函数与导数综合", "代数方程与不等式"],
    "intermediate algebra": ["代数方程与不等式", "数列与递推"],
    "prealgebra": ["代数方程与不等式"],
    "geometry": ["平面几何(圆/相似/共点共线)", "立体几何与空间向量"],
    "number theory": ["初等数论(整除/同余/不定方程)", "数论竞赛题(费马小定理/中国剩余)"],
    "counting & probability": ["排列组合与二项式定理", "概率与期望"],
    "precalculus": ["三角函数与解三角形", "级数求和与极限估算"],
    "gsm8k应用题": ["概率与期望", "代数方程与不等式"],
    "aime竞赛": ["组合数学(计数/图论/博弈)", "整数规划与丢番图方程",
               "数论竞赛题(费马小定理/中国剩余)"],
}
DEFAULT_TOPICS = ["组合数学(计数/图论/博弈)", "整数规划与丢番图方程"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", nargs="+")
    ap.add_argument("--top", type=int, default=6, help="报告最弱考点数")
    args = ap.parse_args()

    agg = {}          # subject -> [err, total]
    lvl = {}          # level -> [err, total]
    for path in args.eval_json:
        d = json.load(open(path))
        for tname, r in d.items():
            for it in r.get("items", []):
                key = (it.get("subject") or "未知").lower()
                e = agg.setdefault(key, [0, 0])
                e[1] += 1
                e[0] += 0 if it["correct"] else 1
                lk = it.get("level", "?")
                l2 = lvl.setdefault(f"{lk}", [0, 0])
                l2[1] += 1
                l2[0] += 0 if it["correct"] else 1

    if not agg:
        print("结果JSON中无items明细(旧版eval产出), 请用新版eval.py重跑")
        return

    print("═" * 60)
    print("考点失分排名 (错误数降权)")
    print("═" * 60)
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    for sub, (err, tot) in rows:
        if tot == 0:
            continue
        print(f"  {sub:<28} 错{err:>3}/{tot:<4}  错误率 {err/tot*100:5.1f}%")
    print("─" * 60)
    print("难度档错误率:")
    for l, (err, tot) in sorted(lvl.items()):
        if tot:
            print(f"  Level {l:<4} 错{err:>3}/{tot:<4}  错误率 {err/tot*100:5.1f}%")

    # 生成SQLM主题权重: 错误数占比 → 主题倍率
    weights = {}
    total_err = sum(e for e, _ in agg.values()) or 1
    for sub, (err, tot) in rows:
        if err == 0 or tot < 10:
            continue
        w = err / total_err
        for topic in SUBJECT2TOPIC.get(sub, DEFAULT_TOPICS):
            weights[topic] = weights.get(topic, 1.0) + w * 4   # 基础1.0+失分加权
    weights = {k: round(v, 2) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])}

    out = weights[0:0]
    with open("weights_for_sqlm.json", "w") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print("═" * 60)
    print("SQLM主题权重(写入 weights_for_sqlm.json):")
    for k, v in weights.items():
        print(f"  {k}: ×{v}")
    print("\n用法: python3 sqlm_gen.py --weights weights_for_sqlm.json ...")

if __name__ == "__main__":
    main()
