"""Coverage and source-validation tests for the exhaustive V3 mapper."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.schemas import QuestionRequest, UsageRecord
from app.v3.exhaustive_mapper import ExhaustiveMapper, _canonical_quote, _pack_records
from app.v3.models import CompiledDocument, CompiledRecord
from app.v3.question_compiler import compile_question


def test_record_packing_preserves_all_records_and_bounds_normal_batches():
    records = [
        CompiledRecord(
            record_id=f"r{index}",
            ordinal=index,
            title=f"Record {index}",
            page_start=index,
            page_end=index,
            anchor_page=index,
            text="x" * size,
        )
        for index, size in enumerate((3000, 3000, 7000, 20_000), start=1)
    ]

    batches = _pack_records(records, max_items=3, max_characters=8000)

    assert [record.record_id for batch in batches for record in batch] == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]
    assert [len(batch) for batch in batches] == [2, 1, 1]
    assert all(
        sum(len(record.text) for record in batch) <= 8000
        for batch in batches
        if len(batch) > 1 or len(batch[0].text) <= 8000
    )


class MapperFakeClient(BaseLLMClient):
    async def chat(self, messages, **kwargs):
        del messages
        results = [
            {
                "question_id": "q1",
                "record_id": "record-001",
                "status": "evidence_found",
                "evidence": [{
                    "claim": "The code name is Aurora.",
                    "exact_quote": "The internal code name was Aurora.",
                    "page": 1,
                    "entity": "Aurora",
                    "field": "code_name",
                    "confidence": 1.0,
                }],
                "topic_assessments": [],
            },
            {
                "question_id": "q1",
                "record_id": "record-002",
                "status": "no_evidence",
                "evidence": [],
                "topic_assessments": [],
            },
        ]
        return LLMResponse(
            content=json.dumps({"results": results}),
            usage=UsageRecord(
                provider="fake",
                model=kwargs["model"],
                stage=kwargs["stage"],
                input_tokens=10,
                output_tokens=10,
            ),
        )

    async def close(self):
        return None


def test_truncated_literal_quote_is_restored_from_source():
    record = CompiledRecord(
        record_id="record-001",
        ordinal=1,
        title="One",
        page_start=9,
        page_end=9,
        anchor_page=9,
        text=(
            "[Page 9]\nStarting as a small igloo-art gallery in 1989, the Icehotel "
            "has grown into a winter extravaganza built of ice blocks. Next sentence."
        ),
    )

    restored = _canonical_quote(
        record,
        "Starting as a small igloo-art gallery in 1989, the Icehotel has grown...",
    )

    assert restored == (
        "Starting as a small igloo-art gallery in 1989, the Icehotel has grown into "
        "a winter extravaganza built of ice blocks.",
        9,
    )


@pytest.mark.asyncio
async def test_mapper_returns_one_validated_result_per_record(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        mistral_api_key="fake",
        record_batch_size=2,
        max_concurrent_requests=1,
        evaluation_mode=False,
    )
    tracker = UsageTracker()
    mapper = ExhaustiveMapper(
        LLMRouter(settings, client=MapperFakeClient()),
        tracker,
        settings,
    )
    document = CompiledDocument(
        document_id="doc",
        records=[
            CompiledRecord(
                record_id="record-001",
                ordinal=1,
                title="One",
                page_start=1,
                page_end=1,
                anchor_page=1,
                text="[Page 1]\nThe internal code name was Aurora.",
            ),
            CompiledRecord(
                record_id="record-002",
                ordinal=2,
                title="Two",
                page_start=2,
                page_end=2,
                anchor_page=2,
                text="[Page 2]\nNo project names are given here.",
            ),
        ],
    )
    plan = compile_question(QuestionRequest(
        question_id="q1",
        question="What was the internal code name?",
        category="needle",
    ))

    results = await mapper.map_document(document, [plan])

    assert len(results) == 2
    assert {result.record_id for result in results} == {"record-001", "record-002"}
    assert results[0].evidence[0].page == 1
    assert results[0].evidence[0].exact_quote == "The internal code name was Aurora."
    assert tracker.total_calls == 1
