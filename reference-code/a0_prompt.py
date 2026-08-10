"""记忆聚合参考实现 · A0 prompt profiles.

Two profiles:

* ``FULL_A0`` — contextual extraction + full Preserve / Structure / Segment
  judgment for the sample. This is the default for the first machine sample.
* ``SHORT_CONTEXTUAL_EXTRACTION`` — shorter prompt that still performs
  contextual extraction, defaults to ``PRESERVE_AS_IS``, and does not run the
  complex segmentation judgment.

Length thresholds live in ``ShortPathConfig`` and are **disabled by default**.
No number here is a global semantic rule; a record's length can never by itself
decide CUT vs PRESERVE. The first machine sample should run every record
through ``FULL_A0`` so the cost-saving short path never pollutes the first
read of model judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class A0PromptProfile(str, Enum):
    """Which A0 prompt to build for a record."""

    FULL_A0 = "FULL_A0"
    SHORT_CONTEXTUAL_EXTRACTION = "SHORT_CONTEXTUAL_EXTRACTION"


@dataclass(frozen=True)
class ShortPathConfig:
    """Config-driven short-path gate. Numbers are engineering budget, not
    semantic rules; ``enabled`` defaults to False so the short bypass is off."""

    enabled: bool = False
    max_chars: int = 0  # 0 => disabled; never a universal CUT/preserve rule


def make_a0_prompt(
    record,
    micro_pane,
    profile: A0PromptProfile = A0PromptProfile.FULL_A0,
) -> str:
    """Build the A0 prompt for the given profile.

    ``record`` is a SourceRecord; ``micro_pane`` is the ObservationMicroUnit
    pane. The SHORT profile still contextualizes but does not ask for the
    full segmentation judgment.
    """
    if profile == A0PromptProfile.SHORT_CONTEXTUAL_EXTRACTION:
        return _short_prompt(record, micro_pane)
    return _full_prompt(record, micro_pane)


def _seen_mu_lines(micro_pane) -> str:
    return "\n".join(
        f"[{u.index}] {u.structural_kind.value} {u.start_offset}:{u.end_offset} {u.text}"
        for u in micro_pane.units
    ) or "(no units — empty or whitespace-only source)"


_FULL_JSON_CONTRACT = """\
Output STRICT JSON with this schema (no markdown, no comments):

{
  "record_frame": {
    "overall_event_or_process": "string | null",
    "container_referent": "string | null",
    "time_frame": "string | null",
    "participant_set": ["string", ...],
    "global_context": "string | null"
  },
  "micro_units": [
    {
      "micro_index": 0,
      "local_anchor_or_referent": "string | null",
      "local_focus": "string | null",
      "active_participants": ["string", ...],
      "local_time": "string | null",
      "salient_terms": ["string", ...],
      "local_predications": ["string", ...],
      "relation_to_record_frame": "string | null",
      "constraint_or_decision": "string | null",
      "appraisal_or_reflection": "string | null"
    }
    // ... one entry per micro-unit index listed below, exactly once each
  ],
  "gate": "PRESERVE_AS_IS | PRESERVE_WITH_INTERNAL_STRUCTURE | SEGMENT",

  // --- required when gate == PRESERVE_AS_IS or PRESERVE_WITH_INTERNAL_STRUCTURE ---
  "local_coherence_basis": {"start": 0, "end": 10, "excerpt": "verbatim source text"}

  // --- required when gate == SEGMENT ---
  // "segment_boundaries": [{"start": 0, "end": 15}, ...]
  // "rupture_basis": [{"start": 0, "end": 5, "excerpt": "verbatim source text"}, ...]

  // --- optional when gate == PRESERVE_WITH_INTERNAL_STRUCTURE ---
  // "internal_components": [
  //   {"type_hint": "FACT_OR_EVENT|DECISION_OR_RULE|APPRAISAL_OR_REFLECTION|STANDING_VIEW|OTHER|UNCLEAR",
  //    "start": 0, "end": 5,
  //    "claim_or_predicate": "string | null",
  //    "relation_to_parent_unit": "string | null"}
  // ]
}

Rules:
- micro_units: provide exactly one entry per micro-unit index listed below (full coverage, each index exactly once).
- local_coherence_basis / rupture_basis: return {start, end, excerpt} — the excerpt MUST be a verbatim slice of the source text at [start:end). Do NOT return source_hash or source_id; those are bound deterministically.
- segment_boundaries: sorted, non-overlapping, non-zero-length, must cover all non-blank content.
- internal_components: only for PRESERVE_WITH_INTERNAL_STRUCTURE — explicit source-backed components, NOT one per micro-unit.
- PRESERVE is the default; only SEGMENT with clear local rupture and source-backed rupture_basis.
- local_coherence_basis must NOT equal global_context (global is not local)."""


def _full_prompt(record, micro_pane) -> str:
    mu_lines = _seen_mu_lines(micro_pane)
    return (
        "Read this memory record and output the JSON described below. "
        "Preserve is the default; only SEGMENT with clear local rupture and "
        "source-backed rupture_basis.\n\n"
        f"RECORD:\n{record.raw_text}\n\n"
        f"MICRO-UNITS (provide one micro_units entry per index):\n{mu_lines}\n\n"
        f"JSON CONTRACT:\n{_FULL_JSON_CONTRACT}\n"
    )


_SHORT_JSON_CONTRACT = """\
Output STRICT JSON with this schema (no markdown, no comments):

{
  "record_frame": {
    "overall_event_or_process": "string | null",
    "container_referent": "string | null",
    "time_frame": "string | null",
    "participant_set": ["string", ...],
    "global_context": "string | null"
  },
  "micro_units": [
    {
      "micro_index": 0,
      "local_anchor_or_referent": "string | null",
      "local_predications": ["string", ...]
    }
  ],
  "gate": "PRESERVE_AS_IS",
  "local_coherence_basis": {"start": 0, "end": 10, "excerpt": "verbatim source text"}
}

Rules:
- Default to PRESERVE_AS_IS; do NOT run complex segmentation judgment.
- micro_units is optional for SHORT profile.
- local_coherence_basis: {start, end, excerpt} — excerpt MUST be a verbatim slice."""


def _short_prompt(record, micro_pane) -> str:
    mu_lines = _seen_mu_lines(micro_pane)
    return (
        "Read this short memory record and extract contextualized features "
        "as JSON. Default to PRESERVE_AS_IS; do not run complex segmentation "
        "judgment.\n\n"
        f"RECORD:\n{record.raw_text}\n\n"
        f"MICRO-UNITS:\n{mu_lines}\n\n"
        f"JSON CONTRACT:\n{_SHORT_JSON_CONTRACT}\n"
    )
