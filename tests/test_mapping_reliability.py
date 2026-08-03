"""Reliability tests for exhaustive evidence mapping."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import LLMResponse
from app.llm.usage_tracker import UsageTracker
from app.mapping.batch_mapper import BatchMapper
from app.mapping.evidence_mapper import EvidenceMapper, _build_cache_key
from app.schemas import (
    ChunkMapResult,
    DocumentChunk,
    ExtractionStatus,
    Operator,
    QueryPlan,
)


def _chunk(chunk_id: str = "chunk-current") -> DocumentChunk:
    chunk = DocumentChunk(
        document_id="doc1",
        chunk_id=chunk_id,
        chunk_index=0,
        section_id="section1",
        section_title="Section",
        page_start=3,
        page_end=4,
        text="Evidence text",
    )
    chunk.compute_hash()
    return chunk


def _plan(question: str = "What is the value?") -> QueryPlan:
    return QueryPlan(
        question_id="q1",
        original_question=question,
        operator=Operator.LOOKUP,
        target_fields=["value"],
    )


def _operator_plan(question_id: str, operator: Operator) -> QueryPlan:
    return QueryPlan(
        question_id=question_id,
        original_question=question_id,
        operator=operator,
        target_fields=["value"],
    )


class _Router:
    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0

    def get_model(self, stage: str) -> str:
        return "ministral-3:latest"

    async def chat(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(content=json.dumps(self.response), usage=None)


def test_cache_key_changes_when_question_changes():
    chunk = _chunk()
    first = _build_cache_key(chunk, [_plan("Question A")], "model", "v2")
    second = _build_cache_key(chunk, [_plan("Question B")], "model", "v2")
    assert first != second


@pytest.mark.asyncio
async def test_failure_is_not_cached(tmp_path):
    router = _Router(error=RuntimeError("temporary outage"))
    mapper = EvidenceMapper(router, UsageTracker(), cache_dir=tmp_path)

    first = await mapper.map_chunk(_chunk(), [_plan()])
    second = await mapper.map_chunk(_chunk(), [_plan()])

    assert first[0].extraction_status == ExtractionStatus.LLM_FAILED
    assert second[0].extraction_status == ExtractionStatus.LLM_FAILED
    assert router.calls == 2
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.asyncio
async def test_mapper_coerces_common_local_model_type_errors(tmp_path):
    router = _Router(response={
        "results": [{
            "question_id": "q1",
            "extraction_status": "no_evidence",
            "evidence_items": [],
            "cross_references": "page 9",
            "conflicts": None,
            "uncertainty_notes": ["weak scan", "check table"],
        }],
    })
    mapper = EvidenceMapper(router, UsageTracker(), cache_dir=tmp_path)

    result = (await mapper.map_chunk(_chunk(), [_plan()]))[0]

    assert result.extraction_status == ExtractionStatus.NO_EVIDENCE
    assert result.cross_references == ["page 9"]
    assert result.uncertainty_notes == "weak scan; check table"


@pytest.mark.asyncio
async def test_mapper_preserves_precise_page_from_page_markers(tmp_path):
    router = _Router(response={
        "results": [{
            "question_id": "q1",
            "extraction_status": "evidence_found",
            "evidence_items": [{
                "entity_name": "Example",
                "field_name": "value",
                "raw_value": "42",
                "page_start": 4,
                "page_end": 4,
            }],
        }],
    })
    mapper = EvidenceMapper(router, UsageTracker(), cache_dir=tmp_path)

    result = (await mapper.map_chunk(_chunk(), [_plan()]))[0]

    assert result.evidence_items[0].page_start == 4
    assert result.evidence_items[0].page_end == 4


@pytest.mark.asyncio
async def test_mapper_canonicalizes_field_and_repairs_bc_year(tmp_path):
    plan = QueryPlan(
        question_id="q1",
        original_question="When?",
        operator=Operator.LOOKUP,
        target_fields=["first recorded eruption year"],
        extraction_fields=[{
            "field_name": "first recorded eruption year",
            "expected_type": "number",
        }],
    )
    router = _Router(response={
        "results": [{
            "question_id": "q1",
            "extraction_status": "evidence_found",
            "evidence_items": [{
                "entity_name": "Etna",
                "field_name": "first recorded eruption year",
                "raw_value": "BC",
                "normalized_value": 0,
                "exact_quote": "The first recorded eruption was in 475 BC.",
                "page_start": 3,
            }],
        }],
    })
    mapper = EvidenceMapper(router, UsageTracker(), cache_dir=tmp_path)

    item = (await mapper.map_chunk(_chunk(), [plan]))[0].evidence_items[0]

    assert item.field_name == "first_recorded_eruption_year"
    assert item.normalized_value == "475 BC"


class _ExplodingMapper:
    async def map_chunk(self, chunk, plans):
        raise RuntimeError("task crashed")


@pytest.mark.asyncio
async def test_batch_mapper_records_task_exceptions_for_every_pair():
    settings = Settings(
        llm_provider="mistral_api",
        mistral_api_key="fake",
        max_retries=1,
        question_batch_size=1,
    )
    mapper = BatchMapper(_ExplodingMapper(), settings)
    chunks = [_chunk("c1"), _chunk("c2")]

    results = await mapper.map_all_chunks(chunks, [_plan()])

    assert len(results) == 2
    assert {result.chunk_id for result in results} == {"c1", "c2"}
    assert all(
        result.extraction_status == ExtractionStatus.LLM_FAILED
        for result in results
    )


class _UncertainThenCompleteMapper:
    def __init__(self):
        self.calls = 0

    async def map_chunk(self, chunk, plans):
        self.calls += 1
        status = (
            ExtractionStatus.UNCERTAIN
            if self.calls == 1
            else ExtractionStatus.NO_EVIDENCE
        )
        return [
            ChunkMapResult(
                chunk_id=chunk.chunk_id,
                question_id=plan.question_id,
                extraction_status=status,
            )
            for plan in plans
        ]


@pytest.mark.asyncio
async def test_uncertain_mapping_is_retried():
    settings = Settings(
        llm_provider="mistral_api",
        mistral_api_key="fake",
        max_retries=1,
        question_batch_size=1,
    )
    raw_mapper = _UncertainThenCompleteMapper()
    mapper = BatchMapper(raw_mapper, settings)

    results = await mapper.map_all_chunks([_chunk("c1")], [_plan()])

    assert raw_mapper.calls == 2
    assert results[0].extraction_status == ExtractionStatus.NO_EVIDENCE


def test_batch_profiles_keep_absence_and_synthesis_small():
    settings = Settings(
        llm_provider="mistral_api",
        mistral_api_key="fake",
        question_batch_size=8,
    )
    mapper = BatchMapper(_ExplodingMapper(), settings)
    plans = [
        *[_operator_plan(f"q{i}", Operator.LOOKUP) for i in range(8)],
        *[_operator_plan(f"a{i}", Operator.ABSENCE) for i in range(4)],
        *[_operator_plan(f"s{i}", Operator.GENERAL_SYNTHESIS) for i in range(3)],
    ]

    batches = mapper._split_plans(plans)

    assert [len(batch) for batch in batches] == [8, 3, 1, 3]
    assert all(
        len(batch) <= 3
        for batch in batches
        if batch[0].operator == Operator.ABSENCE
    )


def test_cross_section_category_uses_small_mapping_batches():
    settings = Settings(
        llm_provider="mistral_api",
        mistral_api_key="fake",
        question_batch_size=8,
    )
    mapper = BatchMapper(_ExplodingMapper(), settings)
    plans = [_operator_plan(f"x{i}", Operator.FILTER_LIST) for i in range(3)]
    for plan in plans:
        plan.category = "cross_section"

    assert [len(batch) for batch in mapper._split_plans(plans)] == [3]
