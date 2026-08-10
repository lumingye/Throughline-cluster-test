"""记忆聚合参考实现 · Stage O: deterministic Observation MicroUnits.

MicroUnits are observation coordinates only, NEVER SemanticUnits. Concretely:

- empty lines are NOT semantic boundaries;
- punctuation is ONLY used to derive deterministic sentence/clause coordinate
  windows — it does NOT declare a semantic boundary;
- length does NOT decide a cut;
- a section heading is NOT automatically a parent.

The output is fully deterministic: same input -> identical output, with a
clear span contract (no overlap / no out-of-bounds), stable segment hashes,
and retained structural-level info (``parent_kind``) alongside the leaf
coordinate kind (SENTENCE / CLAUSE / heading / numbered / field).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from normalization import (
    is_field_line,
    is_numbered_item,
    is_section_heading,
    micro_hash,
    source_hash,
    split_clauses,
    split_sentences,
)
from schema import StructuralKind

_CLAUSE_SEP = re.compile(r"[，,、：:]")


@dataclass(frozen=True)
class ObservationMicroUnit:
    index: int
    start_offset: int
    end_offset: int
    structural_kind: StructuralKind
    segment_hash: str
    text: Optional[str] = None  # normalized slice, for tests / evidence only
    # line-level structural kind (e.g. PARAGRAPH) that a SENTENCE / CLAUSE
    # coordinate belongs to; None for leaf-kind lines.
    parent_kind: Optional[StructuralKind] = None


@dataclass(frozen=True)
class MicroUnitPane:
    """All micro-units for one record, plus the bound text reference."""

    source_hash: str
    normalization_version: str
    offset_unit: str
    units: list  # list[ObservationMicroUnit]
    text: str


def _classify(line: str) -> StructuralKind:
    if is_section_heading(line):
        return StructuralKind.SECTION_HEADING
    if is_numbered_item(line):
        return StructuralKind.NUMBERED_ITEM
    if is_field_line(line):
        return StructuralKind.FIELD_LINE
    return StructuralKind.PARAGRAPH


def build_micro_units(record_text: str, source_id: Optional[str] = None) -> MicroUnitPane:
    """Build deterministic observation coordinates over normalized text.

    Each non-blank line becomes a coordinate window. Prose (paragraph) lines
    are further split into sentence / clause coordinate windows (observations
    only, never semantic boundaries). Structural lines (heading / numbered /
    field) keep their own kind. ``parent_kind`` retains the line-level
    structural kind for sentence / clause leaves.
    """
    from normalization import NORMALIZATION_VERSION, OFFSET_UNIT

    text = record_text
    units: list[ObservationMicroUnit] = []
    idx = 0
    for line_start, line in _iter_lines_with_offsets(text):
        if not line.strip():
            continue  # blank lines are not coordinates and not boundaries
        kind = _classify(line)
        if kind == StructuralKind.PARAGRAPH:
            idx = _append_prose(text, line, line_start, units, idx)
        else:
            end = line_start + len(line)
            units.append(
                ObservationMicroUnit(
                    index=idx,
                    start_offset=line_start,
                    end_offset=end,
                    structural_kind=kind,
                    segment_hash=micro_hash(line),
                    text=line,
                )
            )
            idx += 1
    return MicroUnitPane(
        source_hash=source_hash(text),
        normalization_version=NORMALIZATION_VERSION,
        offset_unit=OFFSET_UNIT,
        units=units,
        text=text,
    )


def _append_prose(text: str, line: str, line_start: int, units: list, idx: int) -> int:
    pos = line_start
    for sent in split_sentences(line):
        if _CLAUSE_SEP.search(sent):
            for cl in split_clauses(sent):
                units.append(
                    ObservationMicroUnit(
                        index=idx,
                        start_offset=pos,
                        end_offset=pos + len(cl),
                        structural_kind=StructuralKind.CLAUSE,
                        segment_hash=micro_hash(cl),
                        text=cl,
                        parent_kind=StructuralKind.PARAGRAPH,
                    )
                )
                pos += len(cl)
                idx += 1
        else:
            units.append(
                ObservationMicroUnit(
                    index=idx,
                    start_offset=pos,
                    end_offset=pos + len(sent),
                    structural_kind=StructuralKind.SENTENCE,
                    segment_hash=micro_hash(sent),
                    text=sent,
                    parent_kind=StructuralKind.PARAGRAPH,
                )
            )
            pos += len(sent)
            idx += 1
    return idx


def _iter_lines_with_offsets(text: str):
    """Yield (start_offset, line_without_trailing_lf) for each logical line.

    The normalization in ``normalize_text`` already strips trailing
    whitespace-per-line and normalizes line endings, so offsets here are
    stable code-point offsets into the normalized record text.
    """
    start = 0
    text_len = len(text)
    while start <= text_len:
        nl = text.find("\n", start)
        if nl == -1:
            line = text[start:]
            if line:
                yield start, line
            break
        line = text[start:nl]
        yield start, line
        start = nl + 1
        if start > text_len:
            break