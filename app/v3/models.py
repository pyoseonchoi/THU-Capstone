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
    compiler_version: str = "v3.3"
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
    # Records a model read in full. For these, a missing metric means the
    # record does not report it, rather than that parsing failed.
    profiled_records: list[str] = Field(default_factory=list)
    unassigned_text: str = ""
    page_count: int = 0
    registry_trusted: bool = False
    registry_signals: dict[str, int | str] = Field(default_factory=dict)
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


class QuestionShape(str, enum.Enum):
    """What a question asks the compiled registry to do.

    These are the operations the deterministic executors already implement,
    named so a model can route a question to one instead of the executors
    recognising the phrasings one evaluation happened to use.
    """

    COUNT_ENTITIES = "count_entities"
    COUNT_BY_THRESHOLD = "count_by_threshold"
    EXTREMUM = "extremum"
    UNIT_OUTLIER = "unit_outlier"
    CLAIM_CONFLICT = "claim_conflict"
    RELATION = "relation"
    DATED_EVENT = "dated_event"
    ABSENCE = "absence"
    CONTENTS_INDEX = "contents_index"
    SYNTHESIS = "synthesis"


class ShapePlan(BaseModel):
    """The shape of a question and the arguments that shape needs."""

    shape: QuestionShape = QuestionShape.SYNTHESIS
    # The compiled field whose values the question measures.
    field: str = ""
    # Registry group values the question names, in the order it names them.
    groups: list[str] = Field(default_factory=list)
    # Threshold filter for COUNT_BY_THRESHOLD.
    comparator: str = ""
    threshold: float | None = None
    # Which end of the range EXTREMUM wants.
    direction: str = ""
    # The population a ranking claim covers, copied from the question.
    claim_scope: str = ""
    # What a DATED_EVENT question asks about, and whose event it is.
    event: str = ""
    subject: str = ""
    # Words a document would use for what the question asks about, which need
    # not be the question's own words. A question about an international
    # border reaches a chapter that says "crosses into Kaliningrad" only
    # through wording like this.
    search_terms: list[str] = Field(default_factory=list)
    # Whether the model was sure enough for Python to act on this.
    confident: bool = False

    @property
    def routes(self) -> bool:
        """Whether this plan may drive a deterministic executor."""
        return self.confident and self.shape != QuestionShape.SYNTHESIS


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
    shape_plan: ShapePlan = Field(default_factory=ShapePlan)


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
