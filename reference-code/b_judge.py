"""记忆聚合参考实现 · Stage B: typed semantic judge.

First screen implements ONLY B (no C). Canonical B truth is a single shared
``anchor_kind`` + ``relation`` pair (e.g. ``ENTITY_OR_PROJECT +
SAME_REFERENT_DIFFERENT_EVENT``):

    anchor_kind: EVENT | PROCESS | STANDING_STATE | ENTITY_OR_PROJECT | NONE | UNCLEAR
    relation:    SAME | STAGE_OF | ASPECT_OF | SAME_REFERENT_DIFFERENT_EVENT | UNRELATED | UNCLEAR

Two independent concepts are kept separate:

- ``typed_edge()`` — whether this is an ACCEPTED typed relation edge that
  adds structure (SAME / STAGE_OF / ASPECT_OF / SAME_REFERENT_DIFFERENT_EVENT).
- ``identity_mergeable()`` — whether this may be treated as an identity merge
  (ONLY SAME). STAGE_OF / ASPECT_OF / SAME_REFERENT_DIFFERENT_EVENT are typed
  edges but must NOT be union-find / identity-merged in read-time grouping.

Evidence for any accepted typed relation must be verifiable source evidence:
the provider supplies ``side + start + end`` on the correct unit, then the
importer binds real ``source_id`` / ``source_hash`` and the canonical excerpt
from the actual SourceRecord slice. Fail-closed records
``relation=UNCLEAR`` (a no-structure unknown), never a hard negative
``UNRELATED``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from normalization import NORMALIZATION_VERSION, OFFSET_UNIT
from schema import AnchorKind, Bivalidity, Relation


@dataclass(frozen=True)
class BasisEvidence:
    """A source-backed evidence span for one side of a B claim.

    ``source_id`` / ``source_hash`` are the REAL record identity (not a unit
    id); ``unit_ref`` names which unit side this evidence points to.
    """

    source_id: str
    source_hash: str
    start: int
    end: int
    unit_ref: str
    # Canonical source excerpt, deterministically sliced by the importer (used
    # for local alias display only, never written to committed reports).
    excerpt: Optional[str] = None
    normalization_version: str = NORMALIZATION_VERSION
    offset_unit: str = OFFSET_UNIT


@dataclass(frozen=True)
class BClaim:
    """A single typed B judgment between two units (shared anchor + relation)."""

    pair_key: str
    unit_a: str
    unit_b: str
    anchor_kind: AnchorKind
    relation: Relation
    evidence: list = field(default_factory=list)  # list[BasisEvidence]
    validity: Bivalidity = Bivalidity.ACCEPTED
    fail_closed_reason: Optional[str] = None

    def typed_edge(self) -> bool:
        """Accepted typed relation edge that adds structure.

        UNRELATED and UNCLEAR add no edge; SAME / STAGE_OF / ASPECT_OF /
        SAME_REFERENT_DIFFERENT_EVENT are accepted typed edges.
        """
        if self.validity != Bivalidity.ACCEPTED:
            return False
        return self.relation not in (Relation.UNRELATED, Relation.UNCLEAR)

    def identity_mergeable(self) -> bool:
        """ONLY SAME may be treated as an identity merge / union-find.

        Other typed relations (STAGE_OF / ASPECT_OF / SAME_REFERENT_DIFFERENT_EVENT)
        are real edges but must never be identity-merged in read-time grouping.
        """
        return self.validity == Bivalidity.ACCEPTED and self.relation == Relation.SAME


def parse_b_response(
    raw: Any,
    pair_key: str,
    unit_a: str,
    unit_b: str,
    *,
    source_a,
    source_b,
    unit_spans_a: Optional[list] = None,
    unit_spans_b: Optional[list] = None,
    unit_source_hash_a: Optional[str] = None,
    unit_source_hash_b: Optional[str] = None,
    strict: bool = True,
) -> BClaim:
    """Validate a raw B response into a BClaim.

    ``source_a`` / ``source_b`` are the two SourceRecords the units belong to;
    ``unit_spans_a`` / ``unit_spans_b`` are the SemanticUnit's source_spans
    (list of SpanLocator). Evidence is verified against them (real source
    id/hash, bound span, correct side, span within unit's source_spans, and a
    canonical local excerpt). Fail-closed records relation=UNCLEAR.

    Coverage is checked by ``side`` ("a" and "b"), NOT by ``source_id`` —
    two units from the same SourceRecord (SEGMENT case) must still get
    evidence on both sides.
    """
    if not isinstance(raw, dict):
        return _closed(pair_key, unit_a, unit_b, "PARSE_FAILURE")

    ak = raw.get("anchor_kind")
    rel = raw.get("relation")

    try:
        ak = AnchorKind(ak)
    except (ValueError, TypeError):
        return _closed(pair_key, unit_a, unit_b, "ANCHOR_SCHEMA")
    try:
        rel = Relation(rel)
    except (ValueError, TypeError):
        return _closed(pair_key, unit_a, unit_b, "RELATION_SCHEMA")

    evidence: list = []
    if rel not in (Relation.UNRELATED, Relation.UNCLEAR):
        if strict:
            ev = raw.get("evidence")
            if not isinstance(ev, list) or not ev:
                return _closed(pair_key, unit_a, unit_b, "EVIDENCE_MISSING")
            seen_sides: set = set()
            for item in ev:
                if not isinstance(item, dict):
                    return _closed(pair_key, unit_a, unit_b, "EVIDENCE_SCHEMA")
                ok, ev_item = _validate_evidence(
                    item, pair_key, unit_a, unit_b, source_a, source_b,
                    unit_spans_a, unit_spans_b,
                    unit_source_hash_a, unit_source_hash_b)
                if not ok:
                    return ev_item
                evidence.append(ev_item)
                seen_sides.add(ev_item._side)
            # Coverage by side, NOT by source_id — same-source SEGMENT pairs
            # must still get evidence on both sides.
            if "a" not in seen_sides or "b" not in seen_sides:
                return _closed(pair_key, unit_a, unit_b, "EVIDENCE_BOTH_SIDES")

    return BClaim(
        pair_key=pair_key,
        unit_a=unit_a,
        unit_b=unit_b,
        anchor_kind=ak,
        relation=rel,
        evidence=evidence,
        validity=Bivalidity.ACCEPTED,
    )


def _validate_evidence(item, pair_key, unit_a, unit_b, source_a, source_b,
                        unit_spans_a=None, unit_spans_b=None,
                        unit_source_hash_a=None, unit_source_hash_b=None):
    """Return (True, BasisEvidence) or (False, fail_closed_claim).

    The provider returns only ``side + start + end`` (semantic judgment plus
    span). The deterministic importer binds the real ``source_id`` /
    ``source_hash`` / norm / offset and canonical excerpt from the side's
    SourceRecord — the LLM never copies provenance metadata or authoritative
    source text. The span must also fall within the
    SemanticUnit's ``source_spans`` (when provided) so that same-source
    segment pairs can't use evidence from a different segment.
    """
    side = item.get("side")
    if side not in ("a", "b"):
        return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_SIDE")
    source = source_a if side == "a" else source_b
    unit_ref = unit_a if side == "a" else unit_b
    unit_spans = unit_spans_a if side == "a" else unit_spans_b
    unit_source_hash = unit_source_hash_a if side == "a" else unit_source_hash_b
    if source is None:
        return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_SOURCE_MISSING")
    if unit_source_hash is not None and unit_source_hash != source.source_hash:
        return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_PROVENANCE")
    start, end = item.get("start"), item.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_SPAN")
    if not (0 <= start < end <= len(source.raw_text)):
        return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_SPAN")
    # When the importer provides unit spans, their complete provenance tuple
    # must match the real SourceRecord before numeric containment is trusted.
    # This prevents an old/misbound resume artifact from lending its offsets to
    # different normalized source content.
    if unit_spans is not None:
        locators = [s for s in unit_spans if s is not None]
        if not locators:
            return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_UNIT_SPAN")
        if any(
            s.source_hash != source.source_hash
            or s.normalization_version != source.normalization_version
            or s.offset_unit != source.offset_unit
            for s in locators
        ):
            return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_PROVENANCE")
        if not any(s.start <= start and end <= s.end for s in locators):
            return False, _closed(pair_key, unit_a, unit_b, "EVIDENCE_UNIT_SPAN")
    # Never trust duplicated provider text. Older saved responses may still
    # contain ``excerpt``; it is intentionally ignored so typographic
    # normalization cannot reject an otherwise valid, source-bound span.
    excerpt = source.slice(start, end)
    # Deterministic binding: real source identity comes from the SourceRecord,
    # never from the provider response.
    ev = BasisEvidence(
        source_id=source.source_id,
        source_hash=source.source_hash,
        start=start,
        end=end,
        unit_ref=unit_ref,
        excerpt=excerpt,
        normalization_version=source.normalization_version,
        offset_unit=source.offset_unit,
    )
    # stash side for coverage check (not persisted on the dataclass)
    object.__setattr__(ev, '_side', side)
    return True, ev


def _closed(pair_key, unit_a, unit_b, reason: str) -> BClaim:
    return BClaim(
        pair_key=pair_key,
        unit_a=unit_a,
        unit_b=unit_b,
        anchor_kind=AnchorKind.UNCLEAR,
        relation=Relation.UNCLEAR,
        validity=Bivalidity.FAIL_CLOSED,
        fail_closed_reason=reason,
        evidence=[],
    )
