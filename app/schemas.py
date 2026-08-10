"""Pydantic data models for the entire FULLSCAN-QA pipeline."""

from __future__ import annotations

import enum
import hashlib
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

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


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


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


# ---------------------------------------------------------------------------
# Reduction models
# ---------------------------------------------------------------------------

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

class PipelineAnswer(BaseModel):
    """Final answer for one question."""
    question_id: str
    final_answer: str = ""
    answer_with_evidence: str = ""
    operator: Optional[Operator] = None
    operation_result: Optional[OperationResult] = None
    coverage: Optional[CoverageReport] = None
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
