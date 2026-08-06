"""Pydantic data models for the entire FULLSCAN-QA pipeline."""

from __future__ import annotations

import enum
import hashlib
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.field_names import canonical_field_name

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Operator(str, enum.Enum):
    LOOKUP = "LOOKUP"
    FILTER_LIST = "FILTER_LIST"
    COUNT = "COUNT"
    FILTER_COUNT_LIST = "FILTER_COUNT_LIST"
    ARGMAX = "ARGMAX"
    ARGMIN = "ARGMIN"
    SUM = "SUM"
    AVERAGE = "AVERAGE"
    DIFFERENCE = "DIFFERENCE"
    PERCENT_CHANGE = "PERCENT_CHANGE"
    COMPARE = "COMPARE"
    ABSENCE = "ABSENCE"
    TEMPORAL = "TEMPORAL"
    MULTI_HOP = "MULTI_HOP"
    GENERAL_SYNTHESIS = "GENERAL_SYNTHESIS"


class ExtractionStatus(str, enum.Enum):
    EVIDENCE_FOUND = "evidence_found"
    NO_EVIDENCE = "no_evidence"
    UNCERTAIN = "uncertain"
    PARSE_FAILED = "parse_failed"
    LLM_FAILED = "llm_failed"


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ClaimStatus(str, enum.Enum):
    SUPPORTED = "SUPPORTED"
    DERIVED = "DERIVED"
    CONFLICTING = "CONFLICTING"
    UNSUPPORTED = "UNSUPPORTED"


class TopicLevel(str, enum.Enum):
    SUBSTANTIVE = "substantive"
    MENTION_ONLY = "mention_only"
    NONE = "none"
    UNCERTAIN = "uncertain"


class PageExtractionStatus(str, enum.Enum):
    OK = "ok"
    LOW_TEXT = "low_text"
    NO_TEXT = "no_text"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Top-level document metadata."""
    document_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    filename: str
    file_hash: str = ""
    page_count: int = 0
    total_characters: int = 0
    parsed_at: Optional[datetime] = None
    chunk_count: int = 0


class PageQualityRecord(BaseModel):
    """Quality assessment for a single page."""
    page_number: int
    character_count: int = 0
    block_count: int = 0
    image_count: int = 0
    extraction_status: PageExtractionStatus = PageExtractionStatus.OK
    low_text_warning: bool = False


class DocumentPage(BaseModel):
    """Extracted content of one page."""
    page_number: int
    text: str = ""
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    quality: Optional[PageQualityRecord] = None


class DocumentSection(BaseModel):
    """A detected structural section."""
    section_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    level: int = 0
    page_start: int = 0
    page_end: int = 0
    parent_section_id: Optional[str] = None


class DocumentChunk(BaseModel):
    """A single chunk for processing."""
    document_id: str
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    chunk_index: int
    section_id: str = ""
    section_title: str = ""
    section_titles: list[str] = Field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    text: str
    previous_context_hint: str = ""
    next_context_hint: str = ""
    content_hash: str = ""
    processing_status: ProcessingStatus = ProcessingStatus.PENDING

    def compute_hash(self) -> str:
        """Compute and set the content hash."""
        self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        return self.content_hash


# ---------------------------------------------------------------------------
# Question / Plan models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """A single question from the user."""
    question_id: str = Field(default_factory=lambda: f"q-{uuid.uuid4().hex[:8]}")
    question: str
    category: str = ""


class QuestionBatch(BaseModel):
    """A batch of questions to process together."""
    questions: list[QuestionRequest]


class Condition(BaseModel):
    """A filter condition for entity selection."""
    field: str = ""
    operator: str = "=="  # >=, <=, ==, !=, >, <, contains, not_contains
    value: str = ""
    unit: str = ""

    @model_validator(mode="before")
    @classmethod
    def _preprocess_condition(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"field": v, "operator": "==", "value": v, "unit": ""}
        if v is None:
            return {"field": "", "operator": "==", "value": "", "unit": ""}
        return v

    @field_validator("field", "operator", "value", "unit", mode="before")
    @classmethod
    def _default_none_str(cls, v: Any) -> str:
        return "" if v is None else str(v)


class ExtractionField(BaseModel):
    """A field the mapper should extract."""
    field_name: str = ""
    description: str = ""
    expected_type: str = "string"  # string, number, date, boolean
    unit: str = ""

    @model_validator(mode="before")
    @classmethod
    def _preprocess_extraction_field(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"field_name": v, "description": "", "expected_type": "string", "unit": ""}
        if v is None:
            return {"field_name": "", "description": "", "expected_type": "string", "unit": ""}
        return v

    @field_validator("field_name", "description", "expected_type", "unit", mode="before")
    @classmethod
    def _default_none_str(cls, v: Any) -> str:
        return "" if v is None else str(v)


class QueryPlan(BaseModel):
    """Structured plan for answering a question."""
    question_id: str
    original_question: str
    category: str = ""
    normalized_question: str = ""
    operator: Operator
    entity_type: str = ""
    target_fields: list[str] = Field(default_factory=list)
    extraction_fields: list[ExtractionField] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    candidate_topics: list[str] = Field(default_factory=list)
    grouping_fields: list[str] = Field(default_factory=list)
    operand_entities: list[str] = Field(default_factory=list)
    return_fields: list[str] = Field(default_factory=list)
    required_coverage: str = "all"  # "all" or percentage
    answer_language: str = "auto"
    ambiguity_notes: list[str] = Field(default_factory=list)
    requires_deterministic_computation: bool = False
    normalized_unit: str = ""

    @field_validator(
        "normalized_question",
        "category",
        "entity_type",
        "required_coverage",
        "answer_language",
        "normalized_unit",
        mode="before",
    )
    @classmethod
    def _default_none_str(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator(
        "target_fields",
        "extraction_fields",
        "conditions",
        "candidate_topics",
        "grouping_fields",
        "operand_entities",
        "return_fields",
        "ambiguity_notes",
        mode="before",
    )
    @classmethod
    def _default_none_list(cls, v: Any) -> list:
        return [] if v is None else (v if isinstance(v, list) else [v])

    @model_validator(mode="after")
    def _canonicalize_field_contract(self) -> "QueryPlan":
        self.target_fields = [
            canonical_field_name(field) for field in self.target_fields if field
        ]
        self.grouping_fields = [
            canonical_field_name(field) for field in self.grouping_fields if field
        ]
        self.return_fields = [
            canonical_field_name(field) for field in self.return_fields if field
        ]
        for field in self.extraction_fields:
            field.field_name = canonical_field_name(field.field_name)
            if (
                field.field_name == "location"
                and "country" in field.description.casefold()
            ):
                field.field_name = "country"
                self.target_fields = [
                    "country" if target == "location" else target
                    for target in self.target_fields
                ]
        for condition in self.conditions:
            condition.field = canonical_field_name(condition.field)
            if (
                condition.field == "location"
                and any(field.field_name == "country" for field in self.extraction_fields)
            ):
                condition.field = "country"
        supported_condition_operators = {
            ">=", "<=", ">", "<", "==", "!=", "contains", "not_contains",
            "starts_with", "ends_with",
        }
        valid_conditions = []
        for condition in self.conditions:
            if condition.operator.strip().casefold() in supported_condition_operators:
                valid_conditions.append(condition)
            else:
                self.ambiguity_notes.append(
                    f"Ignored non-filter condition: {condition.field} "
                    f"{condition.operator} {condition.value}".strip()
                )
        self.conditions = valid_conditions
        extraction_names = {field.field_name for field in self.extraction_fields}
        identity_fields = {"entity", "entity_name", "name", "park_name"}
        for condition in self.conditions:
            if condition.field and condition.field not in extraction_names | identity_fields:
                expected_type = (
                    "number"
                    if condition.operator in {">", ">=", "<", "<="}
                    else "string"
                )
                self.extraction_fields.append(ExtractionField(
                    field_name=condition.field,
                    description=f"Field required by condition: {condition.field}",
                    expected_type=expected_type,
                    unit=condition.unit,
                ))
                extraction_names.add(condition.field)
        question = self.original_question.casefold()
        if any(term in question for term in ("contradict", "inconsisten", "conflict")):
            if "claim" not in extraction_names:
                self.extraction_fields.append(ExtractionField(
                    field_name="claim",
                    description="Factual or superlative claim made by the document",
                    expected_type="string",
                ))
        return self


# ---------------------------------------------------------------------------
# Evidence models
# ---------------------------------------------------------------------------

class TopicAssessment(BaseModel):
    """Assessment of a candidate topic in a chunk."""
    topic: str
    level: TopicLevel
    justification: str = ""


class EvidenceQuote(BaseModel):
    """An exact quote from the source."""
    text: str
    page: int = 0


class EvidenceItem(BaseModel):
    """A single piece of extracted evidence."""
    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    question_id: str
    chunk_id: str
    section_id: str = ""
    page_start: int = 0
    page_end: int = 0
    entity_id: str = ""
    entity_name: str = ""
    field_name: str = ""
    raw_value: str = ""
    normalized_value: Optional[float | str] = None
    unit: str = ""
    claim: str = ""
    exact_quote: str = ""
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""
    extraction_status: ExtractionStatus = ExtractionStatus.EVIDENCE_FOUND

    @field_validator("field_name", mode="before")
    @classmethod
    def _canonicalize_field_name(cls, value: Any) -> str:
        return canonical_field_name("" if value is None else str(value))


class EntityRecord(BaseModel):
    """A deduplicated entity with merged evidence."""
    entity_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    entity_name: str
    normalized_name: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    raw_fields: dict[str, str] = Field(default_factory=dict)
    source_chunks: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _canonicalize_field_maps(self) -> "EntityRecord":
        self.fields = {
            canonical_field_name(key): value for key, value in self.fields.items()
        }
        self.raw_fields = {
            canonical_field_name(key): value for key, value in self.raw_fields.items()
        }
        return self


class ChunkMapResult(BaseModel):
    """Result of mapping one chunk for one question."""
    chunk_id: str
    question_id: str
    extraction_status: ExtractionStatus
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    topic_assessments: list[TopicAssessment] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    uncertainty_notes: str = ""
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Reduction models
# ---------------------------------------------------------------------------

class EvidenceLedger(BaseModel):
    """Collected evidence for a single question."""
    question_id: str
    plan: Optional[QueryPlan] = None
    items: list[EvidenceItem] = Field(default_factory=list)
    entities: list[EntityRecord] = Field(default_factory=list)
    topic_matrix: dict[str, dict[str, str]] = Field(default_factory=dict)
    chunk_statuses: dict[str, ExtractionStatus] = Field(default_factory=dict)
    total_chunks: int = 0
    conflicts: list[str] = Field(default_factory=list)


class OperationResult(BaseModel):
    """Result of a deterministic reduction operation."""
    question_id: str
    operator: Operator
    result_value: Optional[float | str | int] = None
    result_list: list[str] = Field(default_factory=list)
    result_table: list[dict[str, Any]] = Field(default_factory=list)
    included_entities: list[str] = Field(default_factory=list)
    excluded_entities: list[str] = Field(default_factory=list)
    missing_field_entities: list[str] = Field(default_factory=list)
    computation_detail: str = ""
    warnings: list[str] = Field(default_factory=list)
    topic_verdicts: dict[str, str] = Field(default_factory=dict)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification models
# ---------------------------------------------------------------------------

class ClaimVerification(BaseModel):
    """Verification result for a single claim."""
    claim: str
    status: ClaimStatus
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    operation_id: str = ""
    correction: str = ""


class CoverageReport(BaseModel):
    """Coverage diagnostics for a question run."""
    total_pages: int = 0
    parsed_pages: int = 0
    unresolved_pages: int = 0
    total_chunks: int = 0
    successful_mappings: int = 0
    no_evidence_chunks: int = 0
    uncertain_chunks: int = 0
    failed_chunks: int = 0
    detected_entity_count: int = 0
    entities_missing_fields: int = 0
    topic_coverage: dict[str, str] = Field(default_factory=dict)
    coverage_percentage: float = 0.0
    required_percentage: float = 100.0
    requirement_met: bool = False
    mapped_chunks: int = 0
    missing_chunks: int = 0
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return (
            self.requirement_met
            and self.failed_chunks == 0
            and self.uncertain_chunks == 0
            and self.unresolved_pages == 0
        )


# ---------------------------------------------------------------------------
# Usage & tracking models
# ---------------------------------------------------------------------------

class UsageRecord(BaseModel):
    """Token and performance tracking for one LLM call."""
    provider: str = ""
    model: str = ""
    stage: str = ""
    prompt_version: str = ""
    chunk_id: str = ""
    question_ids: list[str] = Field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached: bool = False
    latency_ms: float = 0.0
    retry_count: int = 0
    success: bool = True


# ---------------------------------------------------------------------------
# Pipeline output models
# ---------------------------------------------------------------------------

class EvidenceQuote(BaseModel):
    """A single source-literal quote backing an answer, with its page if known."""
    quote: str
    page: Optional[int] = None


class PipelineAnswer(BaseModel):
    """Final answer for one question."""
    question_id: str
    final_answer: str = ""
    answer_with_evidence: str = ""
    evidence_quotes: list["EvidenceQuote"] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    operator: Optional[Operator] = None
    plan: Optional[QueryPlan] = None
    operation_result: Optional[OperationResult] = None
    coverage: Optional[CoverageReport] = None
    verification: list[ClaimVerification] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PipelineRun(BaseModel):
    """Complete run output."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    document_id: str
    pipeline_mode: str = "ADAPTIVE_HIERARCHICAL_V3"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    answers: list[PipelineAnswer] = Field(default_factory=list)
    usage: list[UsageRecord] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    submission_path: str = ""
