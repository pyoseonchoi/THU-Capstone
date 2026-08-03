"""Tests for section-level synthesis reduction."""

from app.reduction.hierarchical_synthesis import reduce_hierarchical_synthesis
from app.schemas import (
    EvidenceItem,
    EvidenceLedger,
    Operator,
    QueryPlan,
)


def test_hierarchical_synthesis_preserves_sections_and_provenance():
    plan = QueryPlan(
        question_id="q1",
        original_question="Synthesize the document",
        operator=Operator.GENERAL_SYNTHESIS,
    )
    items = [
        EvidenceItem(
            question_id="q1",
            chunk_id="c1",
            section_id="north",
            page_start=2,
            page_end=2,
            entity_name="A",
            field_name="theme",
            claim="Northern claim",
            relevance=0.9,
            confidence=0.9,
        ),
        EvidenceItem(
            question_id="q1",
            chunk_id="c2",
            section_id="south",
            page_start=50,
            page_end=50,
            entity_name="B",
            field_name="theme",
            claim="Southern claim",
            relevance=0.9,
            confidence=0.9,
        ),
    ]
    ledger = EvidenceLedger(
        question_id="q1",
        plan=plan,
        total_chunks=2,
        items=items,
    )

    result = reduce_hierarchical_synthesis(plan, ledger)

    assert [row["section_id"] for row in result.result_table] == ["north", "south"]
    assert len(result.supporting_evidence_ids) == 2
    assert result.source_pages == [2, 50]
