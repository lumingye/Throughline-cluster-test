#!/usr/bin/env python3
"""最小流程真跑一遍：默认不切 → 抽卡 → 按对象归组 → 说不出对象的单独放。

它**故意是笨的**：

    归组只按「对象」的字面做精确匹配。不算向量、不认别名、不做聚类。

笨是有目的的。它给出一条下限：

- 若笨办法排出来的东西已经接近人工样张 → 那些复杂召回通道现在还没有理由；
- 若笨办法明显更差 → **差在哪一眼就能看见**，那才是该修的地方。

不要在看到这一屏之前先去建评测体系（严格性应当与决策错了的代价成正比，
而不是与好奇心成正比）。

⚠ 状态：**假数据通了，真语料未跑。**

用法：
    python min_run.py 快照.jsonl --ids-file ids.txt      # 跑人工样张对应的那批
    python min_run.py 快照.sqlite --sample 20 --seed 0   # 或者随便抽几条

配置只从环境变量读：
    MEMAGG_CHAT_API_BASE / MEMAGG_CHAT_API_KEY / MEMAGG_CHAT_MODEL

⚠ 本脚本**会把选中记录的原文发往你配置的那个对话服务**，也会在终端打印
   原文片段。只在本地看，不要把输出贴到公开的地方。
   key 只从环境变量读，不落盘、不打印、不进缓存。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_explicit_refs import load_rows          # 同一个输入契约：id + 正文


# ── 提示词：把已经定下来的几条限制原样写进去 ──────────────────────────────
#
# 每一条限制都对应一个实验结论，改动前先回去看：
#   - 举证责任在「切」这一侧（保留整条是正常输出，不是失败）
#   - 背景条件（同一天/同一个人/同一条记录）不能当依据，正反都不能
#   - 内容功能变化（事实→规则→感想）不等于边界
#   - 依据必须是原文里能找到的一段字，不能是「我觉得」

PROMPT = """你在整理一个人自己写的记忆库。下面是其中一条记录。

第一个问题不是"怎么切"，而是"要不要切"。**默认不切。**
整条保留是正常且常见的结果，不是失败、不是没找到边界。

只有当记录里存在两段各自都能独立成立的经历，且它们之间说不出具体的局部
联系时，才提出切分。

切分必须给依据，依据要能指到原文的一段字。以下**不构成**依据：
- 同一天、同一个人、同一条记录、同一个项目（这些只是背景，正反都不能用）
- 换行、空行、标点（那是排版，不是边界）
- 内容功能变了（"发生了什么 → 所以定了什么规矩 → 我怎么想"通常是一件事，
  不是三件）

然后对每个单元（不切就是一个）抽这几样：

- 对象：这段主要在讲哪一个具体的人/事/物/项目。写成一个短名词。
  说不出就写 UNCLEAR，不要硬凑，也不要用"生活""感受"这种筐。
- 锚点类型：EVENT（一次发生的事）/ PROCESS（有阶段的过程）/
  STANDING_STATE（一直有效的状态或规则）/ ENTITY_OR_PROJECT（人或项目）/
  UNCLEAR
- 依据：从原文里**原样摘抄**一段字（10~40 字），能支持你写的那个对象。
  必须是原文里逐字存在的，不要改写、不要加省略号。
- 体裁：事件记述 / 规则或决定 / 评价或感想 / 状态描述 / 其他
- 关键词：2~5 个，原文里出现过的词优先

只输出 JSON，不要解释，不要代码块围栏：

{"preserve": true/false,
 "cut_basis": "若 preserve 为 false，说明为什么这里可以切；否则空字符串",
 "units": [{"对象": "...", "锚点类型": "...", "依据": "...",
            "体裁": "...", "关键词": ["...", "..."]}]}

记录原文：
"""


def chat_config(args):
    base = args.chat_base or os.environ.get("MEMAGG_CHAT_API_BASE")
    key = os.environ.get("MEMAGG_CHAT_API_KEY")
    mdl = args.chat_model or os.environ.get("MEMAGG_CHAT_MODEL")
    miss = [n for n, v in (("MEMAGG_CHAT_API_BASE", base),
                           ("MEMAGG_CHAT_API_KEY", key),
                           ("MEMAGG_CHAT_MODEL", mdl)) if not v]
    if miss:
        sys.exit("先设这几个环境变量：" + "、".join(miss))
    return {"base": base, "key": key, "model": mdl}


def ask(cfg, prompt, retries=3):
    url = cfg["base"].rstrip("/") + "/chat/completions"
    body = json.dumps({"model": cfg["model"], "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {cfg['key']}"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                return f"__ERROR__ {e}"
            time.sleep(1.5 * (attempt + 1))


def parse_json(txt):
    """模型经常给多余的围栏或前言，取第一个完整的大括号块。"""
    if not txt or txt.startswith("__ERROR__"):
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── 缓存：同一条记录、同一个模型、同一份提示词，不重复付钱 ────────────────
#
# 键必须由**实际发出去的那个字符串**算出来。曾有一版缓存写入时用未 strip 的
# 文本算键、查询时用 strip 过的，于是永远不命中、每次全量重跑，而且静默——
# 只表现为"怎么又这么慢"。这里只在一个地方算键，避免同样的事。

def cache_key(cfg, prompt):
    h = hashlib.sha256()
    h.update(cfg["model"].encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()


def norm_object(s):
    """归组用的对象名规范化。只做最保守的处理：去空白、去标点、统一大小写。

    **刻意不做**别名合并、不做近义合并、不做向量。那些是后面要单独测的
    召回通道；混进来就分不清"归对了"是靠对象抽得准还是靠合并算法。
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[\s　]+", "", s)
    s = re.sub(r"[，。、！？：；\"'（）()《》\[\]【】,.!?:;]", "", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--table", default="memories")
    ap.add_argument("--id-col")
    ap.add_argument("--text-col")
    ap.add_argument("--ids-file", help="一行一个记录标识；给人工样张对应的那批用")
    ap.add_argument("--sample", type=int, help="没有 ids-file 时随便抽几条")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-base")
    ap.add_argument("--chat-model")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=".min-run-cache.json")
    ap.add_argument("--out", help="把逐条结果写成 json，便于回头核对")
    ap.add_argument("--yes", action="store_true", help="跳过发送确认")
    a = ap.parse_args()

    rows = [(i, t) for i, t in load_rows(a.path, a.table, a.id_col, a.text_col)
            if isinstance(t, str) and t.strip()]
    if not rows:
        sys.exit("没有可用记录。")

    if a.ids_file:
        want = {l.strip() for l in open(a.ids_file, encoding="utf-8") if l.strip()}
        picked = [(i, t) for i, t in rows if i in want]
        missing = want - {i for i, _ in picked}
        if missing:
            print(f"⚠ ids-file 里有 {len(missing)} 个标识在库里找不到，已跳过")
    elif a.sample:
        picked = random.Random(a.seed).sample(rows, min(a.sample, len(rows)))
    else:
        sys.exit("要么 --ids-file，要么 --sample N。不默认跑全库。")

    if not picked:
        sys.exit("选出来是空的。")

    cfg = chat_config(a)
    print(f"  对话服务：{cfg['base']}  模型 {cfg['model']}")
    print(f"  ⚠ 即将把 {len(picked)} 条记忆原文发往该服务（每条一次调用）")
    if not a.yes:
        if input("  继续？[y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("已取消。")

    cache = {}
    if os.path.exists(a.cache):
        try:
            cache = json.load(open(a.cache, encoding="utf-8"))
        except Exception:
            cache = {}

    jobs = [(rid, txt, PROMPT + txt[:4000]) for rid, txt in picked]
    todo = [j for j in jobs if cache_key(cfg, j[2]) not in cache]
    print(f"  缓存命中 {len(jobs) - len(todo)} 条，需要调用 {len(todo)} 条")

    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = {ex.submit(ask, cfg, p): (rid, p) for rid, _, p in todo}
            for n, fu in enumerate(as_completed(futs), 1):
                rid, p = futs[fu]
                cache[cache_key(cfg, p)] = fu.result()
                print(f"\r  已完成 {n}/{len(todo)}", end="", flush=True)
        print(f"\r  调用完成，用时 {time.time()-t0:.0f}s" + " " * 12)
        json.dump(cache, open(a.cache, "w", encoding="utf-8"), ensure_ascii=False)

    # ── 收结果 ────────────────────────────────────────────────────────────
    units, failed, preserved = [], [], 0
    for rid, txt, prompt in jobs:
        got = parse_json(cache.get(cache_key(cfg, prompt)))
        if not got or not isinstance(got.get("units"), list):
            failed.append(rid)
            continue
        if got.get("preserve") is not False:
            preserved += 1
        for u in got["units"]:
            if not isinstance(u, dict):
                continue
            basis = (u.get("依据") or "").strip()
            units.append({
                "record": rid,
                "对象": (u.get("对象") or "UNCLEAR").strip(),
                "锚点类型": (u.get("锚点类型") or "UNCLEAR").strip(),
                "依据": basis,
                # 依据能不能在原文里逐字找到。这是唯一便宜又能抓到编造的检查：
                # 指不回原文的依据，等于没有依据。
                "依据可定位": bool(basis) and basis in txt,
                "体裁": (u.get("体裁") or "").strip(),
                "关键词": u.get("关键词") or [],
                "单元数": len(got["units"]),
            })

    if not units:
        sys.exit("一条都没解析出来。先看看 --out 里模型到底回了什么。")

    # ── 归组：只按对象字面 ────────────────────────────────────────────────
    groups = defaultdict(list)
    for u in units:
        key = norm_object(u["对象"])
        groups[key if key and key != "unclear" else None].append(u)

    unnamed = groups.pop(None, [])
    singles = {k: v for k, v in groups.items() if len(v) == 1}
    real = {k: v for k, v in groups.items() if len(v) > 1}

    # ── 出一屏 ────────────────────────────────────────────────────────────
    W = 66
    print("\n" + "=" * W)
    print(f"  记录 {len(picked)} 条 → 语义单元 {len(units)} 个")
    print(f"  整条保留 {preserved}/{len(picked) - len(failed)} "
          f"（{preserved/max(1, len(picked)-len(failed)):.0%}）")
    ok_basis = sum(1 for u in units if u["依据可定位"])
    print(f"  依据能在原文里逐字找到 {ok_basis}/{len(units)} "
          f"（{ok_basis/len(units):.0%}）")
    if failed:
        print(f"  ⚠ 解析失败 {len(failed)} 条：{', '.join(failed[:6])}")
    print("=" * W)

    print(f"\n【成组的】{len(real)} 组\n")
    for key, members in sorted(real.items(), key=lambda kv: -len(kv[1])):
        kinds = {m["锚点类型"] for m in members}
        recs = {m["record"] for m in members}
        print(f"▸ {members[0]['对象']}　{len(members)} 个单元 / "
              f"{len(recs)} 条记录 / 锚点 {'、'.join(sorted(kinds))}")
        for m in members:
            mark = "" if m["依据可定位"] else "  ⚠依据指不回原文"
            print(f"    · [{m['record']}] {m['体裁']}：{m['依据'][:38]}{mark}")
        print()

    print(f"【只有自己一个的】{len(singles)} 个 —— 这不是失败，只是没有同伴\n")
    for key, (m,) in sorted(singles.items()):
        mark = "" if m["依据可定位"] else "  ⚠依据指不回原文"
        print(f"    · {m['对象']}　[{m['record']}]　{m['体裁']}{mark}")

    print(f"\n【说不出对象】{len(unnamed)} 个\n")
    for m in unnamed:
        mark = "" if m["依据可定位"] else "  ⚠依据指不回原文"
        print(f"    · [{m['record']}] {m['体裁']}：{m['依据'][:38]}{mark}")

    print("\n" + "-" * W)
    print("怎么读这一屏：")
    print("  1. 跟人工样张并排看。**只看差在哪，先不要打分。**")
    print("  2. 「成组的」里若有明显该在一起却没在一起的 —— 那是召回问题，")
    print("     说明对象字面匹配不够，才轮到别名/向量通道上场。")
    print("  3. 「依据指不回原文」占比高 —— 先修抽卡，别急着看归组好不好。")
    print("  4. 整条保留率若接近 100%，说明这一跑跟「根本不切」没有区别；")
    print("     若接近 0%，说明默认不切这条没被执行。两头都要停下来看。")
    print("-" * W)
    print("注意：按对象归组只是**原型组织动作**，不等于关系判定里的同一/阶段，")
    print("      也不等于任何上位层级。它现在只负责把可能有关的摆到一起。")

    if a.out:
        json.dump({"units": units, "failed": failed},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n逐条结果已写入：{a.out}")


if __name__ == "__main__":
    main()
