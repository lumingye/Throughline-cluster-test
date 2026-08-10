"""记忆聚合参考实现 · shared enums and value objects.

These constants are the single source of truth for the A0 gate, B typed
relation, anchor kinds, and provenance fields used across the vNext pipeline.
Keeping them here avoids independently handwritten divergent contracts in the
prompt, schema, and parser.
"""

from __future__ import annotations

from enum import Enum


class Gate(str, Enum):
    """A0 preserve-or-segment gate. Only three states are legal."""

    PRESERVE_AS_IS = "PRESERVE_AS_IS"
    PRESERVE_WITH_INTERNAL_STRUCTURE = "PRESERVE_WITH_INTERNAL_STRUCTURE"
    SEGMENT = "SEGMENT"


class A0Option(str, Enum):
    """Machine error handler for a single A0 result."""

    # Result is coherent and was accepted on its own terms.
    ACCEPTED = "ACCEPTED"
    # The result failed a deterministic check and was fail-closed to
    # PRESERVE_AS_IS without re-running the provider.
    FAIL_CLOSED_PRESERVE = "FAIL_CLOSED_PRESERVE"


class AnchorKind(str, Enum):
    """B anchor kind (independent axis)."""

    EVENT = "EVENT"
    PROCESS = "PROCESS"
    STANDING_STATE = "STANDING_STATE"
    ENTITY_OR_PROJECT = "ENTITY_OR_PROJECT"
    NONE = "NONE"
    UNCLEAR = "UNCLEAR"


class Relation(str, Enum):
    """B typed relation (independent axis). No generic MAYBE/RELATED escape."""

    SAME = "SAME"
    STAGE_OF = "STAGE_OF"
    ASPECT_OF = "ASPECT_OF"
    SAME_REFERENT_DIFFERENT_EVENT = "SAME_REFERENT_DIFFERENT_EVENT"
    UNRELATED = "UNRELATED"
    UNCLEAR = "UNCLEAR"


class SourceProfileKind(str, Enum):
    """Source profile is a risk hint, never semantic evidence."""

    DIARY_OR_REFLECTIVE_NARRATIVE = "DIARY_OR_REFLECTIVE_NARRATIVE"
    HANDOFF_OR_LETTER = "HANDOFF_OR_LETTER"
    PROJECT_OR_WORK_LOG = "PROJECT_OR_WORK_LOG"
    CONVERSATION_DERIVED_MEMORY = "CONVERSATION_DERIVED_MEMORY"
    REFERENCE_OR_NOTE = "REFERENCE_OR_NOTE"
    STRUCTURED_LIST = "STRUCTURED_LIST"
    UNKNOWN = "UNKNOWN"


class SourceProfileSource(str, Enum):
    HOST_METADATA = "HOST_METADATA"
    MODEL_INFERRED = "MODEL_INFERRED"
    UNKNOWN = "UNKNOWN"


class CompositionProvenance(str, Enum):
    """How the outer structure of a record was formed.

    Distinct from ``SourceProfileKind``: profile describes text type / risk,
    provenance describes how the outer structure came to be. MODEL_INFERRED
    must not lower structural protection just because content looks like an
    automatic summary.
    """

    HUMAN_AUTHORED = "HUMAN_AUTHORED"
    MACHINE_COMPOSED = "MACHINE_COMPOSED"
    UNKNOWN = "UNKNOWN"


class StructuralKind(str, Enum):
    """Deterministic observation micro-unit structural kind."""

    SENTENCE = "SENTENCE"
    CLAUSE = "CLAUSE"
    PARAGRAPH = "PARAGRAPH"
    NUMBERED_ITEM = "NUMBERED_ITEM"
    FIELD_LINE = "FIELD_LINE"
    SECTION_HEADING = "SECTION_HEADING"


class ComponentTypeHint(str, Enum):
    """Semantic component type hint (not a boundary declaration)."""

    FACT_OR_EVENT = "FACT_OR_EVENT"
    DECISION_OR_RULE = "DECISION_OR_RULE"
    APPRAISAL_OR_REFLECTION = "APPRAISAL_OR_REFLECTION"
    STANDING_VIEW = "STANDING_VIEW"
    OTHER = "OTHER"
    UNCLEAR = "UNCLEAR"


class RelationToFrame(str, Enum):
    """micro_unit relation_to_record_frame."""

    CONTINUATION = "CONTINUATION"
    ELABORATION = "ELABORATION"
    SEQUENCE = "SEQUENCE"
    CAUSAL = "CAUSAL"
    RESPONSE = "RESPONSE"
    REVISION = "REVISION"
    NEW_SUBTOPIC = "NEW_SUBTOPIC"
    FRAME_LEVEL = "FRAME_LEVEL"
    UNCLEAR = "UNCLEAR"


class FailClosedReason(str, Enum):
    """Machine-readable reason for a fail-closed PRESERVE_AS_IS."""

    SCHEMA_FAILURE = "SCHEMA_FAILURE"
    PARSE_FAILURE = "PARSE_FAILURE"
    SPAN_INVALID = "SPAN_INVALID"
    BASIS_MISSING = "BASIS_MISSING"
    SELF_CONTRADICTORY = "SELF_CONTRADICTORY"
    UNRESOLVED_UNCLEAR = "UNRESOLVED_UNCLEAR"
    GLOBAL_AS_LOCAL_BASIS = "GLOBAL_AS_LOCAL_BASIS"
    UNCLEAR = "UNCLEAR"


class Bivalidity(str, Enum):
    """Per-pair B result handling."""

    ACCEPTED = "ACCEPTED"
    FAIL_CLOSED = "FAIL_CLOSED"