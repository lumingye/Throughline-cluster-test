"""记忆聚合参考实现 · normalization contract and source hashing.

A span is only meaningful relative to a specific ``source_hash`` +
``normalization_version`` + ``offset_unit``. We never store bare ``[start,
end]`` offsets. This module owns the normalization and hashing so that every
call site uses the same rule (mirrors the "one place computes the key" lesson
from an earlier cache bug).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Final

# Version the normalization pipeline. Any change to the rules below (or to
# NORMALIZERS) bumps this constant and invalidates stored offsets.
NORMALIZATION_VERSION: Final[str] = "vnext-norm-1"

# The unit offsets are expressed in. We always operate on code points.
OFFSET_UNIT: Final[str] = "code_point"


def normalize_text(text: str) -> str:
    """Deterministic Unicode / newline normalization.

    Applies, in order:
      1. line-end normalization (CRLF / CR / unicode line separators -> LF)
      2. Unicode NFC
      3. strip trailing whitespace on each line
    This is the exact text that observation offsets refer to.
    """
    if not isinstance(text, str):
        raise TypeError("SourceRecord text must be a str")
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u2028", "\n").replace("\u2029", "\n").replace("\x0b", "\n").replace("\x0c", "\n")
    t = unicodedata.normalize("NFC", t)
    lines = [line.rstrip() for line in t.split("\n")]
    # Join stable; drop a single trailing newline so the last line is not an
    # empty element (keeps offsets unambiguous).
    return "\n".join(lines).rstrip("\n")


def source_hash(text: str) -> str:
    """Stable hash over the normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def micro_hash(text: str) -> str:
    """Stable segment hash for one micro-unit's normalized slice."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SpanLocator:
    """A bound source span.

    Never a bare ``(start, end)``: always carries the source hash,
    normalization version, and offset unit it was computed against.
    """

    source_hash: str
    start: int
    end: int
    normalization_version: str = NORMALIZATION_VERSION
    offset_unit: str = OFFSET_UNIT

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")
        if self.normalization_version != NORMALIZATION_VERSION:
            raise ValueError("span normalization version mismatch")
        if self.offset_unit != OFFSET_UNIT:
            raise ValueError("span offset unit mismatch")

    def as_dict(self) -> dict:
        return {
            "source_hash": self.source_hash,
            "start": self.start,
            "end": self.end,
            "normalization_version": self.normalization_version,
            "offset_unit": self.offset_unit,
        }


# Minimal sentence/clause splitter for the deterministic observation layer.
# This is ONLY for producing observation coordinates — it never declares a
# semantic boundary.
_SENT_END = re.compile(r"(?<=[。！？!?；;])")
_CLAUSE_SPLIT = re.compile(r"[，,、：:]")


def split_sentences(line: str) -> list[str]:
    """Split a single line into sentence-type coordinate windows."""
    parts = [p for p in _SENT_END.split(line) if p]
    return parts or [line]


def split_clauses(sentence: str) -> list[str]:
    """Split one sentence into clause-coordinate windows (observation only).

    Separators (``， , 、 ： :``) are kept attached to the preceding clause so
    that ``''.join(split_clauses(s)) == s`` — clause offsets tile the sentence
    exactly, which is required for stable coordinate spans.
    """
    parts = re.split(r"([，,、：:])", sentence)
    clauses: list[str] = []
    cur = ""
    for tok in parts:
        if re.fullmatch(r"[，,、：:]", tok):
            cur += tok
            clauses.append(cur)
            cur = ""
        else:
            cur += tok
    if cur:
        clauses.append(cur)
    return clauses or [sentence]


def is_blank(line: str) -> bool:
    return not line.strip()


_NUMBERED_ITEM = re.compile(r"^\s*(?:[\d一二三四五六七八九十]+[\u3001\uff0e.]|[-*•])\s+")


def is_numbered_item(line: str) -> bool:
    return bool(_NUMBERED_ITEM.match(line))


_FIELD_LINE = re.compile(r"^\s*[^：:\s]{1,40}\s*[：:]\s*\S")


def is_field_line(line: str) -> bool:
    return bool(_FIELD_LINE.match(line))


_SECTION_HEADING = re.compile(
    r"^\s*(?:[#]{1,6}\s+|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[)）])"
)


def is_section_heading(line: str) -> bool:
    return bool(_SECTION_HEADING.match(line))