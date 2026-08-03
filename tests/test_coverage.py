"""Tests for coverage verification rules."""

from __future__ import annotations

from app.schemas import (
    ChunkMapResult,
    DocumentChunk,
    EntityRecord,
    ExtractionStatus,
    Operator,
    PageExtractionStatus,
    PageQualityRecord,
    QueryPlan,
)
from app.verification.coverage_verifier import verify_coverage


def _make_chunk(chunk_id: str, doc_id: str = "doc1") -> DocumentChunk:
    return DocumentChunk(
        document_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        text="test",
    )


def _make_result(
    chunk_id: str, qid: str, status: ExtractionStatus
) -> ChunkMapResult:
    return ChunkMapResult(
        chunk_id=chunk_id,
        question_id=qid,
        extraction_status=status,
    )


class TestCoverageVerifier:
    def test_full_coverage(self):
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        results = [
            _make_result(f"c{i}", "q1", ExtractionStatus.NO_EVIDENCE)
            for i in range(5)
        ]
        plan = QueryPlan(
            question_id="q1",
            original_question="test",
            operator=Operator.COUNT,
        )
        quality = [
            PageQualityRecord(page_number=1, character_count=100)
        ]

        report = verify_coverage(plan, chunks, results, [], quality)
        assert report.coverage_percentage == 100.0
        assert report.failed_chunks == 0

    def test_failed_chunks_reported(self):
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        results = [
            _make_result("c0", "q1", ExtractionStatus.EVIDENCE_FOUND),
            _make_result("c1", "q1", ExtractionStatus.NO_EVIDENCE),
            _make_result("c2", "q1", ExtractionStatus.LLM_FAILED),
            _make_result("c3", "q1", ExtractionStatus.PARSE_FAILED),
        ]
        plan = QueryPlan(
            question_id="q1",
            original_question="test",
            operator=Operator.ARGMAX,
        )

        report = verify_coverage(plan, chunks, results, [], [])
        assert report.failed_chunks == 2
        assert report.coverage_percentage == 50.0
        assert len(report.warnings) > 0

    def test_missing_entity_fields(self):
        chunks = [_make_chunk("c0")]
        results = [
            _make_result("c0", "q1", ExtractionStatus.EVIDENCE_FOUND),
        ]
        entities = [
            EntityRecord(entity_name="A", fields={"area": 100}),
            EntityRecord(entity_name="B", fields={}),
        ]
        plan = QueryPlan(
            question_id="q1",
            original_question="test",
            operator=Operator.ARGMAX,
            target_fields=["area"],
        )

        report = verify_coverage(plan, chunks, results, entities, [])
        assert report.entities_missing_fields == 1

    def test_unresolved_pages_warned(self):
        chunks = [_make_chunk("c0")]
        results = [_make_result("c0", "q1", ExtractionStatus.NO_EVIDENCE)]
        quality = [
            PageQualityRecord(
                page_number=1,
                extraction_status=PageExtractionStatus.NO_TEXT,
                low_text_warning=True,
            ),
        ]
        plan = QueryPlan(
            question_id="q1",
            original_question="test",
            operator=Operator.COUNT,
        )

        report = verify_coverage(plan, chunks, results, [], quality)
        assert report.unresolved_pages == 1
        assert any("pages" in w.lower() for w in report.warnings)
