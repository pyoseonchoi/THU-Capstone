"""Full pipeline test using a fake LLM client.

Tests the complete pipeline with synthetic fixtures and
deterministic fake responses.
"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.llm.usage_tracker import UsageTracker
from app.reduction.deterministic_reducer import reduce
from app.schemas import (
    ChunkMapResult,
    EvidenceItem,
    EvidenceLedger,
    ExtractionStatus,
    Operator,
    QuestionRequest,
    UsageRecord,
)
from tests.fixtures.synthetic_document import (
    create_argmax_plan,
    create_filter_count_plan,
    create_synthetic_entities,
)


class FakeLLMClient(BaseLLMClient):
    """Deterministic fake LLM client for testing."""

    def __init__(self):
        self.call_log: list[dict] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stage: str = "",
        chunk_id: str = "",
        question_ids: list[str] | None = None,
    ) -> LLMResponse:
        """Return deterministic responses based on stage."""
        self.call_log.append({
            "stage": stage,
            "model": model,
            "chunk_id": chunk_id,
            "question_ids": question_ids,
        })

        if stage == "planner":
            return self._planner_response(messages, question_ids)
        elif stage == "mapper":
            return self._mapper_response(messages, chunk_id, question_ids)
        elif stage == "answer":
            return self._answer_response(messages)
        elif stage == "verifier":
            return self._verifier_response(messages)
        else:
            return LLMResponse(
                content="{}",
                usage=UsageRecord(
                    provider="fake", model=model, stage=stage,
                    input_tokens=100, output_tokens=50,
                ),
            )

    def _planner_response(self, messages, question_ids):
        plan = {
            "question_id": question_ids[0] if question_ids else "q1",
            "original_question": "test",
            "normalized_question": "test",
            "operator": "FILTER_COUNT_LIST",
            "entity_type": "park",
            "target_fields": ["highest_point"],
            "conditions": [{"field": "highest_point", "operator": ">=", "value": "3000"}],
            "return_fields": ["count", "names"],
            "required_coverage": "all",
            "requires_deterministic_computation": True,
        }
        return LLMResponse(
            content=json.dumps(plan),
            usage=UsageRecord(
                provider="fake", model="fake", stage="planner",
                input_tokens=200, output_tokens=100,
            ),
        )

    def _mapper_response(self, messages, chunk_id, question_ids):
        results = []
        for qid in (question_ids or ["q1"]):
            results.append({
                "chunk_id": chunk_id,
                "question_id": qid,
                "extraction_status": "no_evidence",
                "evidence_items": [],
                "topic_assessments": [],
            })
        return LLMResponse(
            content=json.dumps({"results": results}),
            usage=UsageRecord(
                provider="fake", model="fake", stage="mapper",
                input_tokens=300, output_tokens=50,
            ),
        )

    def _answer_response(self, messages):
        return LLMResponse(
            content=json.dumps({
                "final_answer": "Test answer.",
                "answer_with_evidence": "Test answer (page 1).",
            }),
            usage=UsageRecord(
                provider="fake", model="fake", stage="answer",
                input_tokens=200, output_tokens=50,
            ),
        )

    def _verifier_response(self, messages):
        return LLMResponse(
            content=json.dumps({
                "claims": [
                    {
                        "claim": "Test answer.",
                        "status": "SUPPORTED",
                        "supporting_evidence_ids": [],
                    }
                ],
                "overall_status": "PASS",
            }),
            usage=UsageRecord(
                provider="fake", model="fake", stage="verifier",
                input_tokens=150, output_tokens=80,
            ),
        )

    async def close(self) -> None:
        pass


class TestFakeLLMPipeline:
    """Test pipeline components with the fake LLM."""

    def test_filter_count_with_entities(self):
        """FILTER_COUNT_LIST with the synthetic entities."""
        entities = create_synthetic_entities()
        plan = create_filter_count_plan()
        ledger = EvidenceLedger(
            question_id=plan.question_id,
            plan=plan,
            entities=entities,
            total_chunks=8,
            chunk_statuses={
                f"chunk_{i}": ExtractionStatus.EVIDENCE_FOUND
                for i in range(8)
            },
        )
        result = reduce(plan, ledger)

        # Parks with highest_point >= 3000:
        # Yellowstone (3462), Yosemite (3997), Rocky Mountain (4346),
        # Glacier (3190), Denali (6190) = 5
        assert result.result_value == 5
        assert "Yellowstone" in result.result_list
        assert "Yosemite" in result.result_list
        assert "Rocky Mountain" in result.result_list
        assert "Glacier" in result.result_list
        assert "Denali" in result.result_list

    def test_argmax_with_entities(self):
        """ARGMAX on area with the synthetic entities."""
        entities = create_synthetic_entities()
        plan = create_argmax_plan()
        ledger = EvidenceLedger(
            question_id=plan.question_id,
            plan=plan,
            entities=entities,
            total_chunks=8,
            chunk_statuses={
                f"chunk_{i}": ExtractionStatus.EVIDENCE_FOUND
                for i in range(8)
            },
        )
        result = reduce(plan, ledger)

        # Yellowstone has largest area (8983)
        assert result.result_value == 8983
        assert result.result_list == ["Yellowstone"]
        # Denali is missing area field
        assert "Denali" in result.missing_field_entities

    def test_duplicate_entity_detection(self):
        """Test that deduplication handles the same entity in two chunks."""
        from app.reduction.deduplicator import deduplicate_entities

        items = [
            EvidenceItem(
                question_id="q1", chunk_id="c1",
                entity_name="Olympic", field_name="area",
                raw_value="3734", normalized_value=3734,
                page_start=8, page_end=8,
            ),
            EvidenceItem(
                question_id="q1", chunk_id="c2",
                entity_name="Olympic", field_name="highest_point",
                raw_value="2432", normalized_value=2432,
                page_start=8, page_end=8,
            ),
        ]
        entities, decisions = deduplicate_entities(items)

        # Should merge into one entity
        assert len(entities) == 1
        assert entities[0].fields.get("area") == 3734
        assert entities[0].fields.get("highest_point") == 2432

    def test_conflicting_values_preserved(self):
        """Test that conflicting values are preserved, not overwritten."""
        from app.reduction.deduplicator import deduplicate_entities

        items = [
            EvidenceItem(
                question_id="q1", chunk_id="c1",
                entity_name="Glacier", field_name="highest_point",
                raw_value="3190", normalized_value=3190,
                page_start=6, page_end=6,
            ),
            EvidenceItem(
                question_id="q1", chunk_id="c1",
                entity_name="Glacier", field_name="highest_point",
                raw_value="3195", normalized_value=3195,
                page_start=6, page_end=6,
            ),
        ]
        entities, _ = deduplicate_entities(items)

        assert len(entities) == 1
        assert len(entities[0].conflicts) > 0
        assert "3190" in entities[0].conflicts[0]
        assert "3195" in entities[0].conflicts[0]

    def test_token_accounting(self):
        """Test usage tracker aggregation."""
        tracker = UsageTracker()
        tracker.record(UsageRecord(
            provider="fake", model="model-a", stage="planner",
            input_tokens=100, output_tokens=50, latency_ms=200,
        ))
        tracker.record(UsageRecord(
            provider="fake", model="model-b", stage="mapper",
            input_tokens=300, output_tokens=80, latency_ms=500,
        ))
        tracker.record(UsageRecord(
            provider="fake", model="model-b", stage="mapper",
            input_tokens=300, output_tokens=80, latency_ms=450,
            success=False,
        ))

        assert tracker.total_input_tokens == 700
        assert tracker.total_output_tokens == 210
        assert tracker.total_calls == 3
        assert tracker.failed_calls == 1

        by_stage = tracker.aggregate_by_stage()
        assert by_stage["planner"]["calls"] == 1
        assert by_stage["mapper"]["calls"] == 2
        assert by_stage["mapper"]["failures"] == 1

        by_model = tracker.aggregate_by_model()
        assert by_model["model-b"]["calls"] == 2

    def test_no_silent_chunk_skipping(self):
        """Verify that ALL chunks get a status in the map results."""
        from tests.fixtures.synthetic_document import create_synthetic_chunks

        chunks = create_synthetic_chunks()
        # Simulate complete mapping
        all_results = []
        for chunk in chunks:
            all_results.append(ChunkMapResult(
                chunk_id=chunk.chunk_id,
                question_id="q1",
                extraction_status=ExtractionStatus.NO_EVIDENCE,
            ))

        # Every chunk must have a result
        result_chunk_ids = {r.chunk_id for r in all_results}
        chunk_ids = {c.chunk_id for c in chunks}
        assert result_chunk_ids == chunk_ids, "Every chunk must have a mapping result"


@pytest.mark.asyncio
async def test_fake_llm_planner():
    """Test the planner with the fake LLM client."""
    from app.llm.router import LLMRouter
    from app.llm.usage_tracker import UsageTracker
    from app.planning.question_planner import QuestionPlanner

    settings = Settings(
        llm_provider="mistral_api",
        mistral_api_key="fake",
    )
    fake = FakeLLMClient()
    router = LLMRouter(settings, client=fake)
    tracker = UsageTracker()
    planner = QuestionPlanner(router, tracker)

    question = QuestionRequest(question_id="q_test", question="How many parks?")
    plan = await planner.plan(question)

    assert plan.question_id == "q_test"
    assert plan.operator == Operator.FILTER_COUNT_LIST
    assert len(fake.call_log) >= 1
    assert fake.call_log[0]["stage"] == "planner"
    assert tracker.total_calls >= 1
