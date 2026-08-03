"""Tests for exhaustive hybrid routing and fact reuse."""

from app.fact_store import FactStore
from app.schemas import (
    ChunkMapResult,
    EvidenceItem,
    ExtractionStatus,
    Operator,
    QueryPlan,
)
from app.strategy import ExecutionStrategy, operation_needs_exhaustive_repair, route_strategy


def _plan(operator: Operator) -> QueryPlan:
    return QueryPlan(
        question_id="q1",
        original_question="test",
        operator=operator,
        target_fields=["area"],
    )


def test_strategy_router():
    assert route_strategy(_plan(Operator.ARGMAX)) == ExecutionStrategy.STRUCTURED_REDUCE
    assert route_strategy(_plan(Operator.ABSENCE)) == ExecutionStrategy.ABSENCE_SCAN
    assert (
        route_strategy(_plan(Operator.GENERAL_SYNTHESIS))
        == ExecutionStrategy.HIERARCHICAL_SYNTHESIS
    )


def test_category_hint_routes_cross_section_to_hierarchical_completion():
    plan = _plan(Operator.FILTER_LIST)
    plan.category = "cross_section"

    assert route_strategy(plan) == ExecutionStrategy.HIERARCHICAL_SYNTHESIS


def test_empty_argmax_requests_exhaustive_repair():
    assert operation_needs_exhaustive_repair(
        _plan(Operator.ARGMAX),
        result_value=None,
        result_list=[],
        result_table=[],
        missing_count=3,
        entity_count=3,
        evidence_count=3,
    )


def test_fact_store_reuses_matching_canonical_fields():
    item = EvidenceItem(
        question_id="other",
        chunk_id="c1",
        entity_name="Park A",
        field_name="Area",
        raw_value="500 sq km",
        normalized_value=500,
    )
    store = FactStore.from_results([
        ChunkMapResult(
            chunk_id="c1",
            question_id="other",
            extraction_status=ExtractionStatus.EVIDENCE_FOUND,
            evidence_items=[item],
        )
    ])

    selected = store.for_plan(_plan(Operator.ARGMAX), [])

    assert selected == [item]
