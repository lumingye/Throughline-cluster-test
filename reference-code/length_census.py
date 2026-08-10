#!/usr/bin/env python3
"""量一下记录有多长，以及"短到不用检查该不该切"这条能省多少。

对应实验记录 T1。

背景：切分是**可选**的——记得好的库可能根本不用切。而很短的记录几乎不可能
装得下两件独立的事，对它们跑整套边界判断是白花钱。

但这条要成立，得先回答两个数，都不能靠猜：

    1. 门槛定在哪？   → 需要一批"人看过、知道要不要切"的样本
    2. 值不值得做？   → 低于门槛的记录占全库多少。只占 5% 就别做了

⚠ 长度**只能用来把记录踢出检查队列，永远不能用来宣布该切**。
   短 → 可以不查；长 → 不代表该切。方向反过来用，就会系统性制造假断裂。

不调模型、不联网、不写任何东西。只读，只统计。

用法：
    python length_census.py 快照.jsonl
    python length_census.py 快照.sqlite --table memories
    python length_census.py 快照.jsonl --labels 标注.tsv   # 有人工标注时

标注文件格式（tsv，一行一条，第二列 1=需要切 0=不需要切）：
    a1<TAB>0
    a2<TAB>1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_explicit_refs import load_rows          # 同一个输入契约：id + 正文


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * p / 100))))
    return sorted_vals[k]


def est_tokens(chars, ratio):
    return int(chars / ratio)


def advise(vals, n, total, W):
    """指出这个库的文字量堆在哪儿。

    **只看文字量的分布，不看内容。** 所以它给的是"该往哪儿使劲"，
    不是"你该切/不该切"——长不等于该切（一篇连贯的长日记就不该切）。
    """
    p50 = pct(vals, 50)
    # 长记录占记录数多少、占**总字数**多少。后者更要紧：
    # 若一成的记录装着七成的字，那切分的重要性远高于条数给人的印象。
    print("\n文字量堆在哪儿（比条数占比更能说明问题）：\n")
    print(f"    {'门槛':>8}   {'条数占比':>8}   {'字数占比':>8}")
    mass = {}
    for thr in (60, 120, 200, 400):
        long = [v for v in vals if v > thr]
        m = sum(long) / total if total else 0
        mass[thr] = m
        print(f"    {thr:>5} 字以上   {len(long)/n:>7.1%}   {m:>7.1%}")

    print("\n" + "=" * W)
    print("  这个库大致是哪一型（经验判断，不是实测阈值）")
    print("=" * W)
    if p50 <= 60 and mass[200] < 0.5:
        kind = ("A 型 · 写的时候已经切好了",
                "切分基本不是问题，别在它上面花力气。",
                "力气应该花在**合**：召回与判关系。")
    elif mass[200] >= 0.5 and p50 >= 200:
        kind = ("B 型 · 长篇型（日记 / 长叙述）",
                "主要风险是**过切**，不是漏切。",
                "优先保留整条 + 内部可寻址，而不是真的切开。")
    elif mass[200] >= 0.5:
        kind = ("C 型 · 容器型（多数条很短，但字都在少数长条里）",
                "那批长条很可能是容器（交接信、日志、自动合并的产物）。",
                "切分在这个库是真问题，**但只对那一小撮记录**。")
    else:
        kind = ("D 型 · 混合",
                "没有明显的形状，别按单一策略配。",
                "先只对长条开切分，其余走保留。")
    print(f"\n  {kind[0]}")
    print(f"    {kind[1]}")
    print(f"    {kind[2]}")
    print("\n  ⚠ 这一段只看长度，不看内容。**长不等于该切**——")
    print("     一篇连贯的长日记既长又完全不该切。它说的只是「值得看的地方在哪」。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--table", default="memories")
    ap.add_argument("--id-col")
    ap.add_argument("--text-col")
    ap.add_argument("--labels", help="tsv：记录标识<TAB>1=需要切/0=不需要切")
    ap.add_argument("--overhead", type=int, default=450,
                    help="每条固定的提示词开销（token），用来算短记录里有多少是白搭的")
    ap.add_argument("--chars-per-token", type=float, default=1.4,
                    help="中文粗估。**这个数应该用一次真实调用的 usage 字段校准**，"
                         "别一直用默认值")
    a = ap.parse_args()

    rows = [(i, t) for i, t in load_rows(a.path, a.table, a.id_col, a.text_col)
            if isinstance(t, str) and t.strip()]
    if not rows:
        sys.exit("没有可用记录。")

    lens = {i: len(t.strip()) for i, t in rows}
    vals = sorted(lens.values())
    n = len(vals)
    R = a.chars_per_token

    total = sum(vals)
    W = 60
    print("=" * W)
    print(f"  记录总数 {n}　总字数 {total}　平均 {total/n:.0f} 字")
    print(f"  字数分布（括号内是按 {R} 字/token 粗估的 token）：")
    for p in (10, 25, 50, 75, 90, 99):
        v = pct(vals, p)
        print(f"      p{p:<3} {v:>7} 字   （≈{est_tokens(v, R):>6} token）")
    print(f"      最长 {vals[-1]} 字   （≈{est_tokens(vals[-1], R)} token）")
    print("=" * W)

    advise(vals, n, total, W)

    # ── 门槛候选：低于它的占多少、以及那些记录里有多少 token 是固定开销 ──
    print("\n若把「短到不用查该不该切」的门槛定在这里，能覆盖多少：\n")
    print(f"    {'门槛':>8}   {'低于它的记录':>12}   {'占比':>7}   "
          f"{'这些记录里固定开销占比':>22}")
    for thr in (20, 40, 60, 80, 120, 200, 300):
        below = [v for v in vals if v <= thr]
        if not below:
            continue
        share = len(below) / n
        avg_tok = est_tokens(sum(below) / len(below), R)
        waste = a.overhead / (a.overhead + avg_tok) if (a.overhead + avg_tok) else 0
        print(f"    {thr:>6} 字   {len(below):>12}   {share:>6.1%}   {waste:>21.0%}")

    print("\n    最后一列的读法：这些短记录送进去时，输入里有多大比例是提示词本身。")
    print("    比例越高，说明**省法不是跳过调用，是给短记录换一份不带切分规则的短提示词**。")

    # ── 有人工标注时：门槛才有依据 ────────────────────────────────────────
    if a.labels:
        lab = {}
        for line in open(a.labels, encoding="utf-8"):
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1] in ("0", "1"):
                lab[parts[0]] = parts[1] == "1"
        hit = {k: v for k, v in lab.items() if k in lens}
        if not hit:
            sys.exit("\n标注文件里的标识跟库里对不上，检查一下。")

        need = sorted(lens[k] for k, v in hit.items() if v)
        keep = sorted(lens[k] for k, v in hit.items() if not v)
        print("\n" + "=" * W)
        print(f"  带标注的样本 {len(hit)} 条：需要切 {len(need)} / 不需要切 {len(keep)}")
        if need:
            print(f"    需要切的：  最短 {need[0]} 字   中位 {pct(need,50)} 字")
        if keep:
            print(f"    不需要切的：最短 {keep[0]} 字   中位 {pct(keep,50)} 字")
        print("=" * W)

        if not need:
            print("\n  样本里没有一条需要切——**这个样本量还不足以定门槛**，再标一些。")
        else:
            # 安全门槛 = 比"最短的那条需要切的"再短一点。
            # 用最短值而不是分位数：门槛的作用是"低于它一定不用查"，
            # 一条反例就足以推翻它。
            safe = need[0] - 1
            below = [v for v in vals if v <= safe]
            print(f"\n  样本里最短的、需要切的记录是 {need[0]} 字。")
            print(f"  → 安全门槛最多只能定到 {safe} 字"
                  f"（全库有 {len(below)} 条 / {len(below)/n:.1%} 低于它）")
            if len(below) / n < 0.10:
                print("\n  ⚠ 覆盖率低于 10%：**这条优化省不下什么，不建议做。**")
            else:
                print("\n  覆盖率够，值得做。但注意这个门槛只在这批样本上没有反例，")
                print("     换一批语料要重新量。")

        overlap = [v for v in keep if need and v >= need[0]]
        if need and overlap:
            print(f"\n  另外：不需要切的记录里，有 {len(overlap)} 条比"
                  f"「最短的需要切的」还长。")
            print("     这说明**长度和该不该切没有干净的分界**——长度只能当"
                  "「短到不用查」的下界，")
            print("     不能反过来当「长了就该查得更凶」的依据。")

    print("\n" + "-" * W)
    print("注意：本脚本只回答「值不值得做、门槛能定到哪」。")
    print("      它**不判断任何一条记录该不该切**。")
    print(f"      字/token 换算用的是默认 {R}，正式用之前拿一次真实调用的")
    print("      usage 字段校准一下，否则上面的 token 数只是数量级。")
    print("-" * W)


if __name__ == "__main__":
    main()
