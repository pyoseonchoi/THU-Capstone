"""Deterministic execution strategy routing from a validated query plan."""

from __future__ import annotations

import enum

from app.schemas import Operator, QueryPlan


class ExecutionStrategy(str, enum.Enum):
    STRUCTURED_REDUCE = "structured_reduce"
    ABSENCE_SCAN = "absence_scan"
    CLAIM_COMPARISON = "claim_comparison"
    HIERARCHICAL_SYNTHESIS = "hierarchical_synthesis"
    EXHAUSTIVE_LOOKUP = "exhaustive_lookup"


def route_strategy(plan: QueryPlan) -> ExecutionStrategy:
    """Choose a post-map strategy without skipping any document chunk."""
    question = plan.original_question.casefold()
    category = plan.category.strip().casefold().replace("-", "_")
    if category == "contradiction":
        return ExecutionStrategy.CLAIM_COMPARISON
    if category in {"global_synthesis", "cross_section"}:
        return ExecutionStrategy.HIERARCHICAL_SYNTHESIS
    if category == "absence":
        return ExecutionStrategy.ABSENCE_SCAN
    if any(term in question for term in ("contradict", "inconsisten", "conflict")):
        return ExecutionStrategy.CLAIM_COMPARISON
    if plan.operator == Operator.ABSENCE:
        return ExecutionStrategy.ABSENCE_SCAN
    if plan.operator in (Operator.MULTI_HOP, Operator.GENERAL_SYNTHESIS):
        return ExecutionStrategy.HIERARCHICAL_SYNTHESIS
    if plan.operator == Operator.COMPARE:
        return ExecutionStrategy.CLAIM_COMPARISON
    if plan.operator == Operator.LOOKUP:
        return ExecutionStrategy.EXHAUSTIVE_LOOKUP
    return ExecutionStrategy.STRUCTURED_REDUCE


def operation_needs_exhaustive_repair(
    plan: QueryPlan,
    *,
    result_value: object,
    result_list: list[str],
    result_table: list[dict],
    missing_count: int,
    entity_count: int,
    evidence_count: int,
) -> bool:
    """Detect structurally unusable reducer output before prose generation."""
    if evidence_count == 0:
        return False
    if plan.operator in {
        Operator.ARGMAX,
        Operator.ARGMIN,
        Operator.DIFFERENCE,
        Operator.PERCENT_CHANGE,
    }:
        return result_value is None
    if plan.operator == Operator.LOOKUP:
        return result_value in (None, "")
    if plan.operator in {Operator.FILTER_LIST, Operator.FILTER_COUNT_LIST}:
        return not result_list and missing_count >= max(1, entity_count // 2)
    if plan.operator == Operator.COMPARE and result_table:
        populated = sum(
            any(value is not None for key, value in row.items() if key != "entity")
            for row in result_table
        )
        return populated == 0
    return False
