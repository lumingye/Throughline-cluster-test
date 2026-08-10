#!/usr/bin/env python3
"""三名标注者的切点集合：相同 / 嵌套 / 交叉 实算。

对应实验记录 E9。

背景：初稿曾断言「粗粒度者的切点集合**始终**是细粒度者的子集」。
把三份答卷的切点还原到统一单元格后逐对实算，该断言为假——27 个两两配对里
有 3 个交叉，且**全部涉及出题者**。

这段代码很短，但它是那次更正的全部依据。之所以值得单独留着，是因为它演示了
一件事：**一个关于「从不发生」的断言，只需要几行代码就能被证伪；
在写下这种断言之前先跑一次，比事后更正便宜得多。**

度量选择：不用「段数一致率」或「切点位置一致率」——两者都会**低估**实际
一致程度（粒度不同不等于判准不同）。改为检验切点集合是否互为子集。

用法：
    python nesting_check.py

数据说明：
- 单元格 = 所有标注者切点的并集所划分出的最小单位；
- 切点记为「在第 i 个单元之后」；
- A / M / N 是三名标注者，A 同时是出题者（其判断需单独打折）。
"""

import statistics

# 九条材料上三名标注者的切点集合。n = 该材料的单元格数。
data = {
    "1": dict(n=5,  A={1, 2, 3, 4},           M=set(),            N=set()),
    "2": dict(n=6,  A={1, 2, 3},              M={2, 3, 5},        N={1, 2, 3, 4, 5}),
    "3": dict(n=10, A=set(),                  M=set(range(1, 10)), N=set()),
    "4": dict(n=6,  A=set(),                  M=set(),            N=set()),
    "5": dict(n=9,  A={2, 3, 4, 5, 6, 7, 8},  M={4, 5, 6, 8},     N={3, 4, 5, 6, 8}),
    "6": dict(n=6,  A={1, 2, 3, 4, 5},        M={1, 2, 3, 4, 5},  N={1, 2, 3, 4, 5}),
    "7": dict(n=4,  A={1},                    M={1, 3},           N={3}),
    "8": dict(n=4,  A={3},                    M=set(),            N=set()),
    "9": dict(n=5,  A={1, 3, 4},              M={2, 3, 4},        N=set()),
}

pairs = [("A", "M"), ("A", "N"), ("M", "N")]


def rel(x, y):
    if x == y:
        return "相同"
    if x < y or y < x:          # 真子集：粒度不同，不算判准分歧
        return "嵌套"
    if not (x & y):
        return "交叉(无公共切点)"
    return "交叉"


def main():
    print(f"{'条':>3}{'单元':>5}  {'A段':>4}{'M段':>4}{'N段':>4}   A-M      A-N      M-N")
    print("-" * 62)
    cross, nest, same = [], [], []
    for k, d in data.items():
        segs = {a: len(d[a]) + 1 for a in "AMN"}
        row = []
        for x, y in pairs:
            r = rel(d[x], d[y])
            row.append(r)
            rec = (k, f"{x}-{y}")
            (same if r == "相同" else nest if r == "嵌套" else cross).append(rec)
        print(f"{k:>3}{d['n']:>5}  {segs['A']:>4}{segs['M']:>4}{segs['N']:>4}   "
              + "  ".join(f"{r:<7}" for r in row))

    tot = len(pairs) * len(data)
    print("-" * 62)
    print(f"总配对 {tot}：相同 {len(same)} / 嵌套 {len(nest)} / 交叉 {len(cross)}")
    print("\n交叉出现在：")
    for k, p in cross:
        print(f"  第 {k} 条  {p}")

    # 并集为空时记为 1.0：两人都不切 = 完全一致
    print("\nJaccard（切点集合）")
    for x, y in pairs:
        vals = []
        for k, d in data.items():
            u = d[x] | d[y]
            i = d[x] & d[y]
            vals.append(1.0 if not u else len(i) / len(u))
        print(f"  {x}-{y}  逐条 {[round(v, 2) for v in vals]}  均值 {statistics.mean(vals):.3f}")

    print("\n注意：交叉全部涉及 A（出题者）。两名未参与出题的标注者之间零交叉。")
    print("      因此「分歧只是粒度」在独立标注者之间成立，作为一般规律不成立。")


if __name__ == "__main__":
    main()
