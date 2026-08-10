#!/usr/bin/env python3
"""扫一遍记录正文里的显式记录间引用，量它的覆盖率。

对应实验记录 T2。

背景：某个库的结构化关系字段为空，据此曾下过"没有显式关系"的硬结论——
那只对**结构化字段**成立。真实语料里，作者把引用**手写在正文中**
（形如「接 <记录标识>」）。这一路的性质很好：免费、高精度、**不依赖词面
重合**，因此能跨过改名。

本脚本只回答一个问题：**它能覆盖多少。**

不调模型、不算向量、不联网、不写任何东西。只读，只统计。

用法：
    python scan_explicit_refs.py 快照.jsonl
    python scan_explicit_refs.py 快照.sqlite --table memories
    python scan_explicit_refs.py 快照.jsonl --dump refs.tsv   # 顺带导出引用对

输出的三个数：
    coverage      有多少条记录带显式引用
    resolvable    其中有多少能在库内找到被指向的那条
    dangling      指向了库里不存在的标识（可能被删/被合并/写错）

安全：不打印记忆正文，只打印标识与统计。--dump 导出的也只有标识对。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter

# ── 引用的写法 ────────────────────────────────────────────────────────────
#
# 目前只见过「接 <标识>」这一种，但作者手写的东西不会只有一种写法。
# 先按几种可能的前缀抓，抓完把**命中的写法分布**打出来——
# 如果某种写法一次都没命中，说明它不存在，删掉即可；
# 如果有大量记录带引用却没被抓到，那是这张表不全，要补。
#
# 标识形如 16 进制串或纯数字（两种 id 形态在真实库里都出现过）。
#
# ⚠ 前缀分两等，这是第一版的教训：
#
# 第一版把「见 / 同 / 续 / 承」这些**单字**也配上了纯数字 ID，结果
# 「见 2026」「同 200 元」「续 3 天」全部命中——在一个满是日期和数字的
# 语料里，那就是个假阳性发生器。首轮 7 个命中里 5 个 dangling，
# 多半就是这么来的。
#
# 现在：
#   强前缀（明确表示接续）        → 纯数字 ID 也认
#   弱前缀（日常用词，容易撞）    → 只认 16 进制形态的 ID
STRONG_PREFIXES = ["接续", "上接", "下接", "接"]
WEAK_PREFIXES = ["延续", "参见", "续", "承", "见", "同"]

_ID_ANY = r"([0-9a-f]{6,}|\d{2,})"
_ID_HEX = r"([0-9a-f]{6,})"

# 长的写法排在前面：「参见」必须先于「见」、「接续」必须先于「接」匹配，
# 否则同一处会被数两遍（短前缀命中长前缀的一部分）。
REF_PATTERNS = (
    [(p, re.compile(p + r"\s*[:：]?\s*" + _ID_ANY)) for p in STRONG_PREFIXES]
    + [(p, re.compile(p + r"\s*[:：]?\s*" + _ID_HEX)) for p in WEAK_PREFIXES]
)
REF_PATTERNS.sort(key=lambda kv: -len(kv[0]))

# 括号里的写法：（接 xxxx）/ (接 xxxx)。括号本身就是很强的信号，故放宽到任意 ID。
BRACKET = re.compile(
    r"[（(]\s*(?:" + "|".join(STRONG_PREFIXES + WEAK_PREFIXES)
    + r")\s*[:：]?\s*" + _ID_ANY + r"\s*[）)]")


def id_shape(s: str) -> str:
    """给命中的标识分个形状——用来一眼看出假阳性。

    真实库里的 id 是 16 进制串或整数序号；而「20xx」几乎一定是年份，
    两三位的裸数字多半是数量、天数、金额。
    """
    if re.fullmatch(r"[0-9a-f]{6,}", s) and not s.isdigit():
        return "hex（像 id）"
    if re.fullmatch(r"20\d{2}", s):
        return "20xx（多半是年份）"
    if len(s) <= 3:
        return "≤3 位数字（多半是数量/天数/金额）"
    return "长数字（可能是序号 id）"


def find_refs(text: str):
    """返回 [(命中的写法, 被指向的标识), ...]。

    **同一处文字只算一次**：按跨度去重，长写法优先。
    否则「参见」会同时命中「参见」和「见」，「（接…）」会同时命中括号写法和裸写法。
    """
    out, taken = [], []          # taken: 已被占用的字符区间

    def free(s, e):
        return all(e <= a or s >= b for a, b in taken)

    def claim(s, e):
        taken.append((s, e))

    for m in BRACKET.finditer(text):
        if free(*m.span()):
            claim(*m.span())
            out.append(("（括号）", m.group(1)))
    for name, pat in REF_PATTERNS:
        for m in pat.finditer(text):
            if free(*m.span()):
                claim(*m.span())
                out.append((name, m.group(1)))
    return out


# ── 读数据：输入契约只需要 id + 正文 ─────────────────────────────────────

COMMON_ID = ("id", "uuid", "key", "bucket_id", "rowid")
COMMON_TEXT = ("content", "text", "body", "message", "summary")


def _pick(d, cands):
    low = {k.lower(): k for k in d}
    return next((low[c] for c in cands if c in low), None)


def load_rows(path, table, id_col, text_col):
    if not os.path.exists(path):
        sys.exit(f"路径不存在：{path}")
    with open(path, "rb") as f:
        head = f.read(16)

    if head.startswith(b"SQLite format 3\x00"):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
            if not cols:
                sys.exit(f"表 {table} 不存在")
            low = {c.lower(): c for c in cols}
            i = id_col or next((low[c] for c in COMMON_ID if c in low), cols[0])
            t = text_col or next((low[c] for c in COMMON_TEXT if c in low), None)
            if not t:
                sys.exit(f"找不到文字列，用 --text-col 指定。可选：{cols}")
            return [(str(r[0]), r[1] or "") for r in
                    conn.execute(f'SELECT "{i}", "{t}" FROM "{table}"')]
        finally:
            conn.close()

    text = open(path, encoding="utf-8-sig").read()
    try:
        data = json.loads(text)
        rows = data if isinstance(data, list) else next(
            (v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)),
            [data])
    except json.JSONDecodeError:
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]

    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        sys.exit("解不出记录。")
    i = id_col or _pick(rows[0], COMMON_ID)
    t = text_col or _pick(rows[0], COMMON_TEXT)
    if not t:
        sys.exit(f"找不到文字字段，用 --text-col 指定。可选：{list(rows[0])[:12]}")
    return [(str(r.get(i, n)), r.get(t) or "") for n, r in enumerate(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--table", default="memories")
    ap.add_argument("--id-col")
    ap.add_argument("--text-col")
    ap.add_argument("--dump", help="把引用对导出到这个 tsv（只含标识，不含正文）")
    a = ap.parse_args()

    rows = load_rows(a.path, a.table, a.id_col, a.text_col)
    rows = [(i, t) for i, t in rows if isinstance(t, str) and t.strip()]
    ids = {i for i, _ in rows}
    n = len(rows)
    if not n:
        sys.exit("没有可用记录。")

    pairs, by_form, with_ref = [], Counter(), 0
    for rid, txt in rows:
        refs = find_refs(txt)
        if not refs:
            continue
        with_ref += 1
        for form, target in refs:
            by_form[form] += 1
            pairs.append((rid, target))

    # 被指向的标识可能是完整 id，也可能是前缀（作者手写常只写前几位）
    def resolve(target):
        if target in ids:
            return target
        hit = [i for i in ids if i.startswith(target)]
        return hit[0] if len(hit) == 1 else None

    resolved = [(s, resolve(t), t) for s, t in pairs]
    ok = [(s, d) for s, d, _ in resolved if d]
    dangling = [(s, raw) for s, d, raw in resolved if not d]
    self_ref = [(s, d) for s, d in ok if s == d]

    print("=" * 58)
    print(f"  记录总数              {n}")
    print(f"  带显式引用的记录      {with_ref}  （{with_ref/n:.1%}）  ← coverage")
    print(f"  引用对总数            {len(pairs)}")
    print(f"    库内可解析          {len(ok)}"
          + (f"  （{len(ok)/len(pairs):.1%}）" if pairs else ""))
    print(f"    指向不存在的标识    {len(dangling)}")
    if self_ref:
        print(f"    自引用（可疑）      {len(self_ref)}")
    print("=" * 58)

    if by_form:
        print("\n命中的写法分布（没命中的写法说明库里没有，可从表里删掉）：")
        for form, c in by_form.most_common():
            print(f"    {form:<8} {c}")

    # 标识形状——一眼看出假阳性。真 id 应该压倒性地落在 hex 或长数字上；
    # 「20xx」和「≤3 位」占比高，说明正则又把日期/数量当成引用了。
    shapes = Counter(id_shape(t) for _, t in pairs)
    if shapes:
        print("\n命中标识的形状（假阳性会集中在下面两类）：")
        for sh, c in shapes.most_common():
            mark = "  ⚠" if ("年份" in sh or "≤3" in sh) else ""
            print(f"    {sh:<26} {c}{mark}")
        bad = sum(c for sh, c in shapes.items() if "年份" in sh or "≤3" in sh)
        if bad:
            print(f"    → 其中 {bad}/{len(pairs)} 形状可疑，"
                  f"排除后实际引用对约 {len(pairs)-bad}")

    # dangling 的形状单独看一眼：如果 dangling 几乎全是可疑形状，
    # 那它们就不是"历史 ID 失效"，只是正则抓错了。
    if dangling:
        dsh = Counter(id_shape(t) for _, t in dangling)
        print("\n指向不存在标识的那些，形状是：")
        for sh, c in dsh.most_common():
            print(f"    {sh:<26} {c}")
        susp = sum(c for sh, c in dsh.items() if "年份" in sh or "≤3" in sh)
        if susp == len(dangling):
            print("    → 全部是可疑形状：**这些多半是正则假阳性，不是 ID 失效**")
        elif susp:
            print(f"    → {susp}/{len(dangling)} 可疑；其余才值得当作真的 ID 失效查")

    # 连成链看看：引用是不是形成了长链，还是零散的一对一
    if ok:
        out_deg = Counter(s for s, _ in ok)
        in_deg = Counter(d for _, d in ok)
        nodes = set(out_deg) | set(in_deg)
        print(f"\n参与引用的记录 {len(nodes)} 条（{len(nodes)/n:.1%}）")
        print(f"    最长出度 {max(out_deg.values())}   最长入度 {max(in_deg.values())}")
        chain_heads = [x for x in nodes if x not in in_deg]
        print(f"    链头（没人指向它）{len(chain_heads)} 条")

    print("\n" + "-" * 58)
    print("怎么读这三个数：")
    print("  coverage 低         → 这一路只能覆盖一小块，别当主召回")
    print("  可解析比例低        → 引用写法没抓全，或标识经历过迁移/合并")
    print("  参与引用的记录多且成链 → 这一路值得排在所有召回之前")
    print("-" * 58)
    print("注意：本脚本只说『有多少引用』，**不说这些引用意味着什么关系**。")
    print("      「接上一条」可能是同一事件、下一阶段、同项目新任务，或仅是回应——")
    print("      它只配当候选来源（candidate_source），不能直接记成 SAME_EVENT。")

    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            f.write("source_id\ttarget_id\tresolved\n")
            for s, d, raw in resolved:
                f.write(f"{s}\t{raw}\t{d or ''}\n")
        print(f"\n引用对已导出：{a.dump}（只含标识，无正文）")


if __name__ == "__main__":
    main()
