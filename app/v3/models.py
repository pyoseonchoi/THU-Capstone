"""Typed models for the V3 document compiler and adaptive executors."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class Strategy(str, enum.Enum):
    STRUCTURED_REDUCE = "structured_reduce"
    EXHAUSTIVE_LOOKUP = "exhaustive_lookup"
    ABSENCE_MATRIX = "absence_matrix"
    CLAIM_COMPARE = "claim_compare"
    HIERARCHICAL_SYNTHESIS = "hierarchical_synthesis"


class OperationKind(str, enum.Enum):
    """Deterministic operations that may be composed for one question."""

    FILTER = "filter"
    COUNT = "count"
    LIST = "list"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
    JOIN = "join"
    DATE_DIFFERENCE = "date_difference"
    COMPARE = "compare"
    GROUP_BY = "group_by"


class OperationStep(BaseModel):
    """One validated step in a deterministic question execution plan."""

    kind: OperationKind
    field: str = ""
    comparator: str = ""
    value: Any = None
    values: list[Any] = Field(default_factory=list)
    group_by: str = "entity"


class NumberFact(BaseModel):
    """A numeric value bound to its source label before any LLM sees it."""

    record_id: str
    field: str
    label: str
    value: float
    raw_value: str
    unit: str = ""
    subject: str = ""
    page: int
    quote: str


class CompiledRecord(BaseModel):
    """One repeated document record, such as a park chapter."""

    record_id: str
    ordinal: int
    title: str
    country: str = ""
    page_start: int
    page_end: int
    anchor_page: int
    text: str
    number_facts: list[NumberFact] = Field(default_factory=list)


class CompiledTableRow(BaseModel):
    """One normalized table row with its original source location."""

    row_id: str
    label: str
    page: int
    rank: int | None = None
    group: str = ""
    values: dict[str, Any] = Field(default_factory=dict)
    quote: str = ""


class CompiledTable(BaseModel):
    """A consecutive table range reconstructed independently of questions."""

    table_id: str
    number: int | None = None
    identifier: str = ""
    title: str
    page_start: int
    page_end: int
    columns: list[str] = Field(default_factory=list)
    rows: list[CompiledTableRow] = Field(default_factory=list)
    raw_text: str = ""
    trusted: bool = False
    warnings: list[str] = Field(default_factory=list)


class ContentsEntry(BaseModel):
    """One entry from a document contents sub-list."""

    category: str
    identifier: str
    title: str = ""
    page: int | None = None


class CompiledDocument(BaseModel):
    """Question-independent representation produced once per upload."""

    document_id: str
    compiler_version: str = "v3.2"
    record_kind: str = "segments"
    entity_label: str = "record"
    records: list[CompiledRecord] = Field(default_factory=list)
    supplementary_records: list[CompiledRecord] = Field(default_factory=list)
    tables: list[CompiledTable] = Field(default_factory=list)
    contents_entries: list[ContentsEntry] = Field(default_factory=list)
    contents_text: str = ""
    contents_pages: list[int] = Field(default_factory=list)
    contents_trusted: bool = False
    field_catalog: list[str] = Field(default_factory=list)
    unassigned_text: str = ""
    page_count: int = 0
    registry_trusted: bool = False
    registry_signals: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class EvidenceCandidate(BaseModel):
    """Question-specific evidence with exact source provenance."""

    question_id: str
    record_id: str
    claim: str
    exact_quote: str
    page: int
    role: str = "support"
    entity: str = ""
    field: str = ""
    value: Any = None
    unit: str = ""
    confidence: float = 0.0
    record_ordinal: int = 0


class TopicAssessment(BaseModel):
    """Per-record verdict used to prove or reject absence."""

    question_id: str
    record_id: str
    topic: str
    level: str
    exact_quote: str = ""
    page: int = 0


class V3MapResult(BaseModel):
    """Terminal result for one question over one compiled record."""

    question_id: str
    record_id: str
    status: str
    evidence: list[EvidenceCandidate] = Field(default_factory=list)
    topics: list[TopicAssessment] = Field(default_factory=list)
    error: str = ""


class V3QuestionPlan(BaseModel):
    """Small, answer-oriented plan with no free-form arithmetic operator."""

    question_id: str
    question: str
    category: str = ""
    strategy: Strategy
    candidate_topics: list[str] = Field(default_factory=list)
    required_slots: list[str] = Field(default_factory=list)
    operations: list[OperationStep] = Field(default_factory=list)
    target_fields: list[str] = Field(default_factory=list)
    entity_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Answer draft produced by a deterministic or evidence-reduce executor."""

    question_id: str
    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    complete: bool = False
    strategy: Strategy
    warnings: list[str] = Field(default_factory=list)


class EvidencePacket(BaseModel):
    """Deterministically reduced evidence handed to the final answer model."""

    question_id: str
    records_scanned: int = 0
    expected_records: int = 0
    evidence: list[EvidenceCandidate] = Field(default_factory=list)
    topic_summary: dict[str, dict[str, int]] = Field(default_factory=dict)
    complete: bool = False
    warnings: list[str] = Field(default_factory=list)
