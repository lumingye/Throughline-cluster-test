"""记忆聚合参考实现 · deterministic candidate recall and union.

Blocking proposes historical SemanticUnits for a later typed B/C judge. It
never declares identity, relation, or parenthood. Channel scores remain local
to their channel; the union keeps every route's provenance and uses an
explainable deterministic-first, then round-robin delivery budget.

The lexical index is deliberately small and dependency-free for the isolated
first-screen implementation. Embedding and host/direct routes are injected as
ordinary ``ChannelCandidate`` rows; this module does not call a provider.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Iterable, Optional


class CandidateRoute(str, Enum):
    EXPLICIT_RECORD_REFERENCE = "EXPLICIT_RECORD_REFERENCE"
    EXPLICIT_ALIAS_OR_RENAME = "EXPLICIT_ALIAS_OR_RENAME"
    EXACT_REFERENT_OR_NAME = "EXACT_REFERENT_OR_NAME"
    HOST_TAG = "HOST_TAG"
    LEXICAL_BM25 = "LEXICAL_BM25"
    RAW_EMBEDDING = "RAW_EMBEDDING"
    CONTEXTUALIZED_EMBEDDING = "CONTEXTUALIZED_EMBEDDING"
    TEMPORAL_OR_STRUCTURAL = "TEMPORAL_OR_STRUCTURAL"
    GRAPH_NEIGHBOR = "GRAPH_NEIGHBOR"


DIRECT_ROUTES = (
    CandidateRoute.EXPLICIT_RECORD_REFERENCE,
    CandidateRoute.EXPLICIT_ALIAS_OR_RENAME,
    CandidateRoute.EXACT_REFERENT_OR_NAME,
    CandidateRoute.HOST_TAG,
)

SOFT_ROUTE_ORDER = (
    CandidateRoute.LEXICAL_BM25,
    CandidateRoute.RAW_EMBEDDING,
    CandidateRoute.CONTEXTUALIZED_EMBEDDING,
    CandidateRoute.TEMPORAL_OR_STRUCTURAL,
    CandidateRoute.GRAPH_NEIGHBOR,
)


@dataclass(frozen=True)
class CandidateSource:
    """One channel-local reason that a target was proposed."""

    route: CandidateRoute
    rank: Optional[int] = None
    score: Optional[float] = None
    basis: Any = None
    index_version: Optional[str] = None
    extractor_version: Optional[str] = None
    model_version: Optional[str] = None


@dataclass(frozen=True)
class ChannelCandidate:
    """A single route's candidate row before cross-route union.

    ``query_ordinal`` and ``target_ordinal`` are ingestion sequence numbers.
    Requiring target < query makes online-causal behavior explicit for injected
    routes as well as for the local lexical index.
    """

    query_unit_id: str
    target_unit_id: str
    query_ordinal: int
    target_ordinal: int
    source: CandidateSource


@dataclass(frozen=True)
class Candidate:
    query_unit_id: str
    target_unit_id: str
    candidate_sources: tuple[CandidateSource, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BlockingResult:
    query_unit_id: str
    candidates: tuple[Candidate, ...]
    pre_cap_count: int
    budget_dropped: int

    def pairs(self) -> list[tuple[str, str]]:
        return [(self.query_unit_id, c.target_unit_id) for c in self.candidates]


def union_candidates(
    query_unit_id: str,
    rows: Iterable[ChannelCandidate],
    *,
    budget: int = 12,
) -> BlockingResult:
    """Dedupe route rows and apply an explainable delivery budget.

    Direct routes are delivered first in fixed route order and are never
    trimmed by the soft-candidate budget. Remaining routes are round-robin
    interleaved by their channel-local rank until that budget is filled. No
    global score is computed and scores from different channels are never
    compared.
    """
    if budget < 0:
        raise ValueError("candidate budget must be >= 0")

    by_target: dict[str, list[CandidateSource]] = defaultdict(list)
    best_route_target: dict[tuple[CandidateRoute, str], CandidateSource] = {}
    for row in rows:
        if row.query_unit_id != query_unit_id:
            raise ValueError("candidate query_unit_id does not match union query")
        if row.target_unit_id == query_unit_id:
            raise ValueError("blocking candidate cannot target the query itself")
        if row.target_ordinal >= row.query_ordinal:
            raise ValueError("non-causal candidate: target must predate query")
        key = (row.source.route, row.target_unit_id)
        current = best_route_target.get(key)
        if current is None or _source_sort_key(row.source) < _source_sort_key(current):
            best_route_target[key] = row.source

    for (_, target), source in best_route_target.items():
        by_target[target].append(source)

    for sources in by_target.values():
        sources.sort(key=_source_sort_key)

    selected: list[str] = []
    selected_set: set[str] = set()

    for route in DIRECT_ROUTES:
        targets = _targets_for_route(by_target, route)
        for target in targets:
            if target not in selected_set:
                selected.append(target)
                selected_set.add(target)

    # The configured budget is a minimum delivery allowance for candidates,
    # not a cap that may discard direct hits.  If direct hits already exceed
    # it, deliver them all and admit no additional soft candidates.
    delivery_limit = max(budget, len(selected))
    soft_queues = {
        route: _targets_for_route(by_target, route)
        for route in SOFT_ROUTE_ORDER
    }
    positions = {route: 0 for route in SOFT_ROUTE_ORDER}
    while len(selected) < delivery_limit:
        progressed = False
        for route in SOFT_ROUTE_ORDER:
            queue = soft_queues[route]
            while positions[route] < len(queue):
                target = queue[positions[route]]
                positions[route] += 1
                if target in selected_set:
                    continue
                selected.append(target)
                selected_set.add(target)
                progressed = True
                break
            if len(selected) >= delivery_limit:
                break
        if not progressed:
            break

    candidates = tuple(
        Candidate(
            query_unit_id=query_unit_id,
            target_unit_id=target,
            candidate_sources=tuple(by_target[target]),
        )
        for target in selected
    )
    return BlockingResult(
        query_unit_id=query_unit_id,
        candidates=candidates,
        pre_cap_count=len(by_target),
        budget_dropped=max(0, len(by_target) - len(candidates)),
    )


def _targets_for_route(
    by_target: dict[str, list[CandidateSource]],
    route: CandidateRoute,
) -> list[str]:
    rows = []
    for target, sources in by_target.items():
        matching = [s for s in sources if s.route == route]
        if matching:
            best = min(matching, key=_source_sort_key)
            rows.append((best.rank is None, best.rank or 0, target))
    rows.sort()
    return [target for _, _, target in rows]


def _source_sort_key(source: CandidateSource) -> tuple:
    route_order = DIRECT_ROUTES + SOFT_ROUTE_ORDER
    try:
        route_idx = route_order.index(source.route)
    except ValueError:
        route_idx = len(route_order)
    return (route_idx, source.rank is None, source.rank or 0)


@dataclass(frozen=True)
class _LexicalEntry:
    unit: Any
    source: Any
    ordinal: int
    tokens: tuple[str, ...]


class LexicalCandidateIndex:
    """Incremental BM25 candidate route over source-backed unit text."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._entries: list[_LexicalEntry] = []

    def add(self, unit, source, *, ordinal: int) -> None:
        if unit.source_id != source.source_id or unit.source_hash != source.source_hash:
            raise ValueError("unit/source provenance mismatch")
        self._entries.append(
            _LexicalEntry(
                unit=unit,
                source=source,
                ordinal=ordinal,
                tokens=tuple(_tokens(_retrieval_text(unit, source))),
            )
        )

    def query(self, unit, source, *, ordinal: int, top_k: int = 5) -> list[ChannelCandidate]:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if unit.source_id != source.source_id or unit.source_hash != source.source_hash:
            raise ValueError("unit/source provenance mismatch")
        query_tokens = set(_tokens(_retrieval_text(unit, source)))
        history = [e for e in self._entries if e.ordinal < ordinal]
        if not query_tokens or not history or top_k == 0:
            return []

        doc_freq = Counter()
        for entry in history:
            doc_freq.update(set(entry.tokens))
        avg_len = sum(len(e.tokens) for e in history) / len(history)
        scored = []
        for entry in history:
            if entry.unit.unit_id == unit.unit_id:
                continue
            score = _bm25_score(
                query_tokens,
                Counter(entry.tokens),
                doc_freq,
                len(history),
                avg_len,
                self.k1,
                self.b,
            )
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].unit.unit_id))

        rows = []
        for rank, (score, entry) in enumerate(scored[:top_k], start=1):
            rows.append(
                ChannelCandidate(
                    query_unit_id=unit.unit_id,
                    target_unit_id=entry.unit.unit_id,
                    query_ordinal=ordinal,
                    target_ordinal=entry.ordinal,
                    source=CandidateSource(
                        route=CandidateRoute.LEXICAL_BM25,
                        rank=rank,
                        score=score,
                        basis={"matched_terms": sorted(query_tokens & set(entry.tokens))},
                        index_version="reference:bm25-v1",
                        extractor_version="reference:retrieval-text-v1",
                    ),
                )
            )
        return rows


def _retrieval_text(unit, source) -> str:
    parts = []
    for span in unit.source_spans:
        if span is not None:
            parts.append(source.slice(span.start, span.end))
    if unit.anchor_or_referent:
        parts.append(unit.anchor_or_referent)
    parts.extend(str(p) for p in unit.predications)
    parts.extend(str(t) for t in unit.features.get("salient_terms", []))
    for component in unit.components:
        if component.claim_or_predicate:
            parts.append(component.claim_or_predicate)
    return "\n".join(parts)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _bm25_score(query_tokens, term_freq, doc_freq, n_docs, avg_len, k1, b) -> float:
    score = 0.0
    doc_len = sum(term_freq.values())
    norm = k1 * (1.0 - b + b * doc_len / (avg_len or 1.0))
    for token in query_tokens:
        tf = term_freq.get(token, 0)
        if not tf:
            continue
        df = doc_freq[token]
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        score += idf * (tf * (k1 + 1.0)) / (tf + norm)
    return score
