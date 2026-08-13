"""离线审计报告的自校验器（对应补遗 S5）。

要防的是一件很具体的事：**审计结论被人手写进报告**。

一份审计报告同时包含逐行观测和一个总结论。如果结论是作者写上去的，
那么「结论」与「数据」之间没有任何约束——数据可以说 A，结论可以写 B，
而且没人会发现。这个校验器把总结论变成**从行数据机械推出来的量**，
再要求报告里声明的那个值与推出来的值相等。

它检查四件事：

1. **判定规则**在看数据之前就写死，且写在代码里而不是报告里；
2. **报告声明的分类 == 从行数据推出的分类**（不等则失败）；
3. **语义摘要自洽**——报告去掉摘要字段后重新哈希，必须等于报告里记的那个；
4. **泄漏扫描**——报告文本里不得出现私有定位符或载荷字段名。

第 3 条的作用是让报告变得不可静默编辑：改任何一个数字，摘要就对不上。

脱敏说明：原版里的基线提交号、提示词摘要、私有运行目录已全部移除，
禁用词表改为通用形态。判定规则与推导逻辑与原版一致。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# 行结构。审计行是定长元组，字段名单独存一份，避免每行重复写 key。
# ---------------------------------------------------------------------------

ROW_SCHEMA = (
    "card_id",              # 判别卡编号
    "polarity",             # POSITIVE | CONTROL
    "view",                 # 同一批材料的不同视图（如 原始 / 已切分）
    "repeat",               # 第几次重复
    "identity_recognized",  # 模型是否识别到共享所指：True | False | UNAVAILABLE
    "hypothesis_count",     # 被提出并记录的多成员假设数
    "terminal_decision",    # 终止裁决
    "terminal_evidence_class",
    "detail_difference_seen",
    "decisive_distinctness_seen",
    "evidence_refs_complete",
    "parser_accepted",
)

_IDX = {name: i for i, name in enumerate(ROW_SCHEMA)}


def _field(row: list[Any], name: str) -> Any:
    return row[_IDX[name]]


# ---------------------------------------------------------------------------
# 判定规则。**在看到数据之前冻结**——这是整个校验器唯一有价值的地方。
# ---------------------------------------------------------------------------

CLASSIFICATION_RESULT = {
    "A": "SHARED_REFERENT_NOT_PROPOSED",
    "B": "SHARED_REFERENT_PROPOSED_BUT_WRONGLY_REJECTED",
    "C": "MIXED_OR_INSUFFICIENT",
}

RULE = (
    "B 要求：每张正例卡都有 识别到所指 == True、支持成员数 > 0、"
    "终止拒绝发生在聚合阶段，且对照卡的主因主要是「所指不同」。"
    "A 要求：正例上根本没有候选假设被提出，且没有识别到所指。"
    "其余一律 C。"
)


def derive_classification(
    rows: Iterable[list[Any]],
    fields_by_card: dict[str, dict[str, Any]],
    positive_cards: set[str],
) -> str:
    """从行数据推出分类。不读报告里声明的那个值。"""
    rows = list(rows)
    control_cards = set(fields_by_card) - positive_cards

    # 哪些卡真的提出过多成员假设（>= 2 个成员才算一个假设）
    proposed = {
        card
        for card in fields_by_card
        if any(
            isinstance(_field(r, "hypothesis_count"), int)
            and _field(r, "hypothesis_count") >= 2
            for r in rows
            if _field(r, "card_id") == card
        )
    }

    positive_identity = all(
        fields_by_card[c]["referent_identity_recognized"] is True for c in positive_cards
    )
    positive_support = all(
        isinstance(fields_by_card[c]["supporting_member_count"], int)
        and fields_by_card[c]["supporting_member_count"] > 0
        for c in positive_cards
    )
    positive_terminal_at_aggregation = all(
        fields_by_card[c]["terminal_rejection_stage"] == "AGGREGATION"
        for c in positive_cards
    )
    controls_distinct = all(
        fields_by_card[c]["primary_rejection_reason"] == "DIFFERENT_REFERENT"
        for c in control_cards
    )

    if (
        positive_identity
        and positive_support
        and positive_terminal_at_aggregation
        and controls_distinct
        and positive_cards <= proposed
    ):
        return "B"
    if not (positive_cards & proposed) and not positive_identity:
        return "A"
    return "C"


# ---------------------------------------------------------------------------
# 语义摘要。去掉摘要字段本身，其余按稳定序列化后哈希。
# ---------------------------------------------------------------------------

DIGEST_FIELD = "semantic_digest"


def semantic_digest(report: dict[str, Any]) -> str:
    payload = {k: v for k, v in report.items() if k != DIGEST_FIELD}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 泄漏扫描。审计的对象是私有运行产物，报告本身必须是可公开的。
# ---------------------------------------------------------------------------

FORBIDDEN_TOKENS = (
    "api_key",
    "authorization",
    "system_prompt",
    "user_payload",
    "raw_provider_response",
    # 真实用法里这里还要列上私有运行目录的绝对路径前缀。
    # 脱敏版不含任何真实路径。
)


def scan_for_leaks(*texts: str) -> list[str]:
    blob = "\n".join(texts).lower()
    return [tok for tok in FORBIDDEN_TOKENS if tok in blob]


# ---------------------------------------------------------------------------
# 主校验
# ---------------------------------------------------------------------------

REQUIRED_PER_CARD_FIELDS = (
    "referent_identity_recognized",
    "supporting_member_count",
    "contradicting_member_count",
    "unresolved_member_count",
    "terminal_rejection_stage",
    "supporting_subset_distinctness",
    "distinctness_outside_supporting_subset",
    "primary_rejection_reason",
    "repeat_reason_consistent",
    "view_reason_consistent",
)


def verify(report_path: Path, prose_path: Path | None = None) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["audit_rows"]
    fields_by_card = report["required_fields_by_card"]

    # 合规前提：这次审计确实没有花钱、没有写私有源
    assert report["no_provider_calls"] is True
    assert report["private_source_zero_write"] is True
    assert report["schema_implementation_authorized"] is False

    # 覆盖面：卡 × 视图 × 重复，一行都不能少
    expected_rows = report["counts"]["cards"] * report["counts"]["views"] * report["counts"]["repeats"]
    assert len(rows) == expected_rows, f"缺行：{len(rows)} != {expected_rows}"
    assert all(len(r) == len(ROW_SCHEMA) for r in rows)

    # 每张卡的必填字段一个都不能缺。**允许取值是「不可得」，不允许字段缺席**——
    # 「不可得」本身就是这次审计最重要的观测结果。
    for card, values in fields_by_card.items():
        missing = set(REQUIRED_PER_CARD_FIELDS) - set(values)
        assert not missing, f"{card} 缺字段：{sorted(missing)}"

    # 解析层必须全部接受，否则测的是解析失败而不是语义
    assert all(_field(r, "parser_accepted") for r in rows)
    assert all(_field(r, "evidence_refs_complete") for r in rows)

    # ★ 核心：声明值必须等于推导值
    positive = {c for c in fields_by_card if any(
        _field(r, "card_id") == c and _field(r, "polarity") == "POSITIVE" for r in rows
    )}
    derived = derive_classification(rows, fields_by_card, positive)
    assert report["classification"] == derived, (
        f"报告声明 {report['classification']}，但从行数据推出的是 {derived}"
    )
    assert report["private_audit_result"] == CLASSIFICATION_RESULT[derived]
    assert report["mechanical_derivation"]["derived_classification"] == derived

    # 摘要自洽：改任何一个数字，这里就会失败
    assert report[DIGEST_FIELD] == semantic_digest(report), "语义摘要对不上：报告被改过"

    # 泄漏扫描
    texts = [report_path.read_text(encoding="utf-8")]
    if prose_path is not None and prose_path.exists():
        texts.append(prose_path.read_text(encoding="utf-8"))
    leaks = scan_for_leaks(*texts)
    assert not leaks, f"报告里出现了不该出现的东西：{leaks}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python audit_verify.py <审计报告.json> [审计正文.md]")
        raise SystemExit(2)
    report_file = Path(sys.argv[1])
    prose_file = Path(sys.argv[2]) if len(sys.argv) > 2 else report_file.with_suffix(".md")
    verify(report_file, prose_file)
    print("审计报告自校验通过：分类可由行数据机械推出，摘要自洽，无泄漏。")
