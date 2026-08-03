"""Deterministic reducer: all operator implementations in Python.

LLMs are never used for counting, arithmetic, or comparisons.
This module receives structured evidence and produces deterministic results.
"""

from __future__ import annotations

from typing import Any, Optional

from app.exceptions import UnsupportedOperatorError
from app.logging_config import get_logger
from app.reduction.normalizer import parse_number
from app.schemas import (
    EntityRecord,
    EvidenceLedger,
    ExtractionStatus,
    OperationResult,
    Operator,
    QueryPlan,
    TopicLevel,
)

logger = get_logger("reduction.deterministic_reducer")


def reduce(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Execute the deterministic operation specified by the query plan.

    Routes to the correct operator implementation and returns a
    structured OperationResult.
    """
    op = plan.operator
    dispatch = {
        Operator.LOOKUP: _reduce_lookup,
        Operator.FILTER_LIST: _reduce_filter_list,
        Operator.COUNT: _reduce_count,
        Operator.FILTER_COUNT_LIST: _reduce_filter_count_list,
        Operator.ARGMAX: _reduce_argmax,
        Operator.ARGMIN: _reduce_argmin,
        Operator.SUM: _reduce_sum,
        Operator.AVERAGE: _reduce_average,
        Operator.DIFFERENCE: _reduce_difference,
        Operator.PERCENT_CHANGE: _reduce_percent_change,
        Operator.COMPARE: _reduce_compare,
        Operator.ABSENCE: _reduce_absence,
        Operator.TEMPORAL: _reduce_temporal,
        Operator.MULTI_HOP: _reduce_synthesis,
        Operator.GENERAL_SYNTHESIS: _reduce_synthesis,
    }

    handler = dispatch.get(op)
    if handler is None:
        raise UnsupportedOperatorError(op.value)

    result = handler(plan, ledger)
    result.question_id = plan.question_id
    result.operator = op
    included = {name.casefold() for name in result.included_entities}
    supporting_entities = [
        entity
        for entity in ledger.entities
        if not included or entity.entity_name.casefold() in included
    ]
    result.supporting_evidence_ids = list(dict.fromkeys(
        evidence_id
        for entity in supporting_entities
        for evidence_id in entity.evidence_ids
    ))
    result.source_pages = sorted({
        page
        for entity in supporting_entities
        for page in entity.source_pages
    })
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_target_field(plan: QueryPlan) -> str:
    """Get the operation field, preferring declared numeric extractions."""
    numeric_operators = {
        Operator.ARGMAX,
        Operator.ARGMIN,
        Operator.SUM,
        Operator.AVERAGE,
        Operator.DIFFERENCE,
        Operator.PERCENT_CHANGE,
    }
    if plan.operator in numeric_operators:
        numeric_fields = [
            field.field_name
            for field in plan.extraction_fields
            if field.expected_type.strip().casefold()
            in {"number", "numeric", "integer", "float"}
        ]
        if numeric_fields:
            return numeric_fields[0]
    if plan.target_fields:
        return plan.target_fields[0]
    return ""


def _get_numeric_value(entity: EntityRecord, field: str) -> Optional[float]:
    """Extract a numeric value for a field from an entity."""
    val = entity.fields.get(field)
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return parse_number(val)
    return None


def _apply_conditions(
    entities: list[EntityRecord],
    conditions: list,
    target_field: str,
) -> tuple[list[EntityRecord], list[str], list[str]]:
    """Filter entities by conditions.

    Returns:
        (matching, excluded_names, missing_field_names)
    """
    matching: list[EntityRecord] = []
    excluded: list[str] = []
    missing: list[str] = []

    if not conditions:
        return list(entities), excluded, missing

    grouped_conditions: dict[str, list] = {}
    for condition in conditions:
        field = condition.field or target_field
        grouped_conditions.setdefault(field, []).append(condition)

    for entity in entities:
        passes = True
        entity_missing = False
        for field, field_conditions in grouped_conditions.items():
            raw_value = (
                entity.entity_name
                if field in ("entity", "entity_name", "name", "park_name")
                else entity.fields.get(field)
            )
            if raw_value is None:
                passes = False
                if entity.entity_name not in missing:
                    missing.append(entity.entity_name)
                entity_missing = True
                break

            outcomes: list[bool] = []
            operators: list[str] = []
            for cond in field_conditions:
                operator = cond.operator.strip().lower()
                operators.append(operator)
                expected = cond.value
                actual_number = _get_numeric_value(entity, field)
                expected_number = parse_number(expected)

                if (
                    operator in (">=", "<=", ">", "<")
                    and actual_number is not None
                    and expected_number is not None
                ):
                    comparisons = {
                        ">=": actual_number >= expected_number,
                        "<=": actual_number <= expected_number,
                        ">": actual_number > expected_number,
                        "<": actual_number < expected_number,
                    }
                    outcomes.append(comparisons[operator])
                elif operator in ("==", "!="):
                    if actual_number is not None and expected_number is not None:
                        equal = actual_number == expected_number
                    else:
                        equal = (
                            str(raw_value).strip().casefold()
                            == expected.strip().casefold()
                        )
                    outcomes.append(equal if operator == "==" else not equal)
                else:
                    actual_text = str(raw_value).casefold()
                    expected_text = expected.casefold()
                    text_checks = {
                        "contains": expected_text in actual_text,
                        "not_contains": expected_text not in actual_text,
                        "starts_with": actual_text.startswith(expected_text),
                        "ends_with": actual_text.endswith(expected_text),
                    }
                    outcomes.append(text_checks.get(operator, False))

            alternatives = len(outcomes) > 1 and all(
                operator in {"==", "contains", "starts_with", "ends_with"}
                for operator in operators
            )
            group_passes = any(outcomes) if alternatives else all(outcomes)
            if not group_passes:
                passes = False
                break

        if passes:
            matching.append(entity)
        elif not entity_missing:
            excluded.append(entity.entity_name)

    return matching, excluded, missing


def _outlier_field(plan: QueryPlan) -> str:
    """Return the compared field for explicit one-versus-all outlier questions."""
    question = plan.original_question.casefold()
    asks_outlier = "different unit" in question and (
        "all the others" in question or "all others" in question
    )
    if not asks_outlier:
        return ""
    if plan.conditions:
        return plan.conditions[0].field
    return next(
        (
            field.field_name
            for field in plan.extraction_fields
            if "unit" in field.field_name
        ),
        "",
    )


def _ordered_operands(
    plan: QueryPlan,
    entities: list[EntityRecord],
) -> list[EntityRecord]:
    """Resolve arithmetic operands by explicit names when available."""
    if not plan.operand_entities:
        return entities
    by_name = {entity.entity_name.strip().casefold(): entity for entity in entities}
    return [
        by_name[name.strip().casefold()]
        for name in plan.operand_entities
        if name.strip().casefold() in by_name
    ]


# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------

def _reduce_lookup(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Simple lookup: return the value of a field for an entity."""
    result = OperationResult(question_id=plan.question_id, operator=plan.operator)
    if ledger.entities:
        entity = ledger.entities[0]
        field = _get_target_field(plan)
        val = entity.fields.get(field, "")
        result.result_value = val
        result.included_entities = [entity.entity_name]
        result.computation_detail = f"Looked up '{field}' for '{entity.entity_name}'"
    return result


def _reduce_filter_list(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Filter entities by conditions and return their names."""
    field = _get_target_field(plan)
    outlier_field = _outlier_field(plan)
    if outlier_field:
        values: dict[str, list[EntityRecord]] = {}
        for entity in ledger.entities:
            value = entity.fields.get(outlier_field)
            if value not in (None, ""):
                values.setdefault(str(value).strip().casefold(), []).append(entity)
        if len(values) >= 2:
            modal_count = max(len(group) for group in values.values())
            matching = [
                entity
                for group in values.values()
                if len(group) < modal_count
                for entity in group
            ]
            excluded = [
                entity.entity_name
                for group in values.values()
                if len(group) == modal_count
                for entity in group
            ]
            return OperationResult(
                question_id=plan.question_id,
                operator=plan.operator,
                result_list=[entity.entity_name for entity in matching],
                result_table=[
                    {
                        "entity": entity.entity_name,
                        outlier_field: entity.fields.get(outlier_field),
                    }
                    for entity in matching
                ],
                included_entities=[entity.entity_name for entity in matching],
                excluded_entities=excluded,
                computation_detail=(
                    f"Selected {len(matching)} entities whose '{outlier_field}' "
                    "differs from the modal value"
                ),
            )
    matching, excluded, missing = _apply_conditions(
        ledger.entities, plan.conditions, field
    )
    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        result_list=[e.entity_name for e in matching],
        included_entities=[e.entity_name for e in matching],
        excluded_entities=excluded,
        missing_field_entities=missing,
    )
    if missing:
        result.warnings.append(
            f"{len(missing)} entities missing field '{field}': {missing}"
        )
    result.computation_detail = (
        f"Filtered {len(ledger.entities)} entities by {plan.conditions}, "
        f"{len(matching)} passed"
    )
    return result


def _reduce_count(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Count unique entities matching conditions."""
    field = _get_target_field(plan)
    if plan.conditions:
        matching, excluded, missing = _apply_conditions(
            ledger.entities, plan.conditions, field
        )
    else:
        matching = ledger.entities
        excluded = []
        missing = []

    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        result_value=len(matching),
        included_entities=[e.entity_name for e in matching],
        excluded_entities=excluded,
        missing_field_entities=missing,
    )
    result.computation_detail = (
        f"Counted {len(matching)} entities out of {len(ledger.entities)}"
    )
    return result


def _reduce_filter_count_list(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Filter, count, and list entities."""
    field = _get_target_field(plan)
    matching, excluded, missing = _apply_conditions(
        ledger.entities, plan.conditions, field
    )

    result_table: list[dict[str, Any]] = [{
        "metric": "total_entities",
        "count": len(ledger.entities),
    }]
    equality_groups: dict[str, list] = {}
    for condition in plan.conditions:
        if condition.operator.strip() == "==":
            equality_groups.setdefault(condition.field or field, []).append(condition)
    for group_field, conditions in equality_groups.items():
        for condition in conditions:
            group_entities, _, _ = _apply_conditions(
                ledger.entities,
                [condition],
                group_field,
            )
            result_table.append({
                "metric": "condition_group",
                "field": group_field,
                "value": condition.value,
                "count": len(group_entities),
                "entities": [entity.entity_name for entity in group_entities],
            })

    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        result_value=len(matching),
        result_list=[e.entity_name for e in matching],
        result_table=result_table,
        included_entities=[e.entity_name for e in matching],
        excluded_entities=excluded,
        missing_field_entities=missing,
    )
    if missing:
        result.warnings.append(
            f"{len(missing)} entities missing field '{field}': {missing}"
        )
    result.computation_detail = (
        f"Filtered by '{field}' conditions, "
        f"{len(matching)}/{len(ledger.entities)} passed"
    )
    return result


def _reduce_argmax(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Find entity with the largest value for a field."""
    field = _get_target_field(plan)
    best_entity: Optional[EntityRecord] = None
    best_value: Optional[float] = None
    missing: list[str] = []

    for entity in ledger.entities:
        val = _get_numeric_value(entity, field)
        if val is None:
            missing.append(entity.entity_name)
            continue
        if best_value is None or val > best_value:
            best_value = val
            best_entity = entity

    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        missing_field_entities=missing,
    )

    if best_entity is not None and best_value is not None:
        result.result_value = best_value
        result.result_list = [best_entity.entity_name]
        result.included_entities = [best_entity.entity_name]
        result.computation_detail = (
            f"ARGMAX('{field}'): '{best_entity.entity_name}' = {best_value} "
            f"(compared {len(ledger.entities) - len(missing)} entities)"
        )
    else:
        result.warnings.append(f"No entities have numeric values for '{field}'")

    if missing:
        result.warnings.append(
            f"{len(missing)} entities missing '{field}': {missing}"
        )
    return result


def _reduce_argmin(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Find entity with the smallest value for a field."""
    field = _get_target_field(plan)
    best_entity: Optional[EntityRecord] = None
    best_value: Optional[float] = None
    missing: list[str] = []

    for entity in ledger.entities:
        val = _get_numeric_value(entity, field)
        if val is None:
            missing.append(entity.entity_name)
            continue
        if best_value is None or val < best_value:
            best_value = val
            best_entity = entity

    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        missing_field_entities=missing,
    )

    if best_entity is not None and best_value is not None:
        result.result_value = best_value
        result.result_list = [best_entity.entity_name]
        result.included_entities = [best_entity.entity_name]
        result.computation_detail = (
            f"ARGMIN('{field}'): '{best_entity.entity_name}' = {best_value} "
            f"(compared {len(ledger.entities) - len(missing)} entities)"
        )
    else:
        result.warnings.append(f"No entities have numeric values for '{field}'")

    if missing:
        result.warnings.append(
            f"{len(missing)} entities missing '{field}': {missing}"
        )
    return result


def _reduce_sum(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Sum numeric values across entities."""
    field = _get_target_field(plan)
    total = 0.0
    included: list[str] = []
    missing: list[str] = []

    for entity in ledger.entities:
        val = _get_numeric_value(entity, field)
        if val is None:
            missing.append(entity.entity_name)
            continue
        total += val
        included.append(entity.entity_name)

    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        result_value=total,
        included_entities=included,
        missing_field_entities=missing,
        computation_detail=f"SUM('{field}') = {total} across {len(included)} entities",
    )
    if missing:
        result.warnings.append(f"{len(missing)} entities excluded (missing field)")
    return result


def _reduce_average(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Average numeric values across entities."""
    field = _get_target_field(plan)
    values: list[float] = []
    included: list[str] = []
    missing: list[str] = []

    for entity in ledger.entities:
        val = _get_numeric_value(entity, field)
        if val is None:
            missing.append(entity.entity_name)
            continue
        values.append(val)
        included.append(entity.entity_name)

    avg = sum(values) / len(values) if values else 0.0

    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        result_value=avg,
        included_entities=included,
        missing_field_entities=missing,
        computation_detail=(
            f"AVG('{field}') = {avg:.4f} across {len(included)} entities"
        ),
    )
    if missing:
        result.warnings.append(f"{len(missing)} entities excluded (missing field)")
    return result


def _reduce_difference(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Compute difference between two entity values."""
    field = _get_target_field(plan)
    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
    )

    operands = _ordered_operands(plan, ledger.entities)
    if len(operands) < 2:
        result.warnings.append("Need at least 2 entities for DIFFERENCE")
        return result

    val_a = _get_numeric_value(operands[0], field)
    val_b = _get_numeric_value(operands[1], field)

    if val_a is not None and val_b is not None:
        diff = val_a - val_b
        result.result_value = diff
        result.included_entities = [
            operands[0].entity_name,
            operands[1].entity_name,
        ]
        result.computation_detail = (
            f"DIFFERENCE: {val_a} - {val_b} = {diff} "
            f"('{operands[0].entity_name}' - '{operands[1].entity_name}')"
        )
    else:
        result.warnings.append("Missing numeric values for difference operands")

    return result


def _reduce_percent_change(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Compute percentage change between two values."""
    field = _get_target_field(plan)
    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
    )

    operands = _ordered_operands(plan, ledger.entities)
    if len(operands) < 2:
        result.warnings.append("Need at least 2 entities for PERCENT_CHANGE")
        return result

    val_old = _get_numeric_value(operands[0], field)
    val_new = _get_numeric_value(operands[1], field)

    if val_old is not None and val_new is not None:
        if val_old == 0:
            result.warnings.append("Zero denominator in percent change")
            result.result_value = None
        else:
            pct = ((val_new - val_old) / abs(val_old)) * 100.0
            result.result_value = round(pct, 4)
            result.computation_detail = (
                f"PERCENT_CHANGE: ({val_new} - {val_old}) / |{val_old}| * 100 = {pct:.4f}%"
            )
        result.included_entities = [
            operands[0].entity_name,
            operands[1].entity_name,
        ]
    else:
        result.warnings.append("Missing values for percent change")

    return result


def _reduce_compare(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Build a structured comparison table."""
    fields = plan.target_fields or [_get_target_field(plan)]
    fields = [f for f in fields if f]

    table: list[dict[str, Any]] = []
    for entity in ledger.entities:
        row: dict[str, Any] = {"entity": entity.entity_name}
        for f in fields:
            row[f] = entity.fields.get(f, None)
        table.append(row)

    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        result_table=table,
        included_entities=[e.entity_name for e in ledger.entities],
        computation_detail=f"Comparison table with {len(table)} entities, fields: {fields}",
    )
    return result


def _reduce_absence(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Determine which candidate topics are absent.

    Rules:
    - substantive anywhere → present
    - mention_only does NOT automatically count as substantive
    - A topic is absent ONLY if:
      1. All chunks reached a terminal status
      2. No chunk marked substantive
      3. No unresolved or uncertain chunks for the topic
    """
    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
    )

    # Check for unresolved/uncertain chunks
    uncertain_chunks = [
        cid for cid, status in ledger.chunk_statuses.items()
        if status in (ExtractionStatus.UNCERTAIN, ExtractionStatus.LLM_FAILED)
    ]
    failed_chunks = [
        cid for cid, status in ledger.chunk_statuses.items()
        if status == ExtractionStatus.PARSE_FAILED
    ]

    absent_topics: list[str] = []
    present_topics: list[str] = []
    mention_only_topics: list[str] = []
    uncertain_topics: list[str] = []

    for topic in plan.candidate_topics:
        topic_assessments = ledger.topic_matrix.get(topic, {})

        # Check if any chunk has substantive coverage
        has_substantive = any(
            level == TopicLevel.SUBSTANTIVE.value
            for level in topic_assessments.values()
        )
        has_mention = any(
            level == TopicLevel.MENTION_ONLY.value
            for level in topic_assessments.values()
        )
        has_uncertain = any(
            level == TopicLevel.UNCERTAIN.value
            for level in topic_assessments.values()
        )

        if has_substantive:
            present_topics.append(topic)
            result.topic_verdicts[topic] = "present_substantive"
        elif uncertain_chunks or has_uncertain:
            uncertain_topics.append(topic)
            result.topic_verdicts[topic] = "uncertain_incomplete_coverage"
        elif has_mention:
            mention_only_topics.append(topic)
            result.topic_verdicts[topic] = "mention_only"
        else:
            # Check all chunks were processed
            processed = len(ledger.chunk_statuses)
            if processed >= ledger.total_chunks and not failed_chunks:
                absent_topics.append(topic)
                result.topic_verdicts[topic] = "absent"
            else:
                uncertain_topics.append(topic)
                result.topic_verdicts[topic] = "uncertain_incomplete_coverage"

    result.result_list = absent_topics
    result.included_entities = present_topics

    if uncertain_topics:
        result.warnings.append(
            f"Cannot confirm absence for topics with incomplete coverage: {uncertain_topics}"
        )
    if failed_chunks:
        result.warnings.append(
            f"{len(failed_chunks)} chunks failed processing — absence determination incomplete"
        )

    result.computation_detail = (
        f"ABSENCE analysis: {len(absent_topics)} absent, "
        f"{len(present_topics)} present, "
        f"{len(mention_only_topics)} mention-only, "
        f"{len(uncertain_topics)} uncertain"
    )

    return result


def _reduce_temporal(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Sort entities by date and identify temporal ordering."""
    field = _get_target_field(plan)
    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
    )

    dated_entities: list[tuple[str, str]] = []
    missing: list[str] = []

    for entity in ledger.entities:
        date_val = entity.fields.get(field)
        if date_val:
            dated_entities.append((entity.entity_name, str(date_val)))
        else:
            missing.append(entity.entity_name)

    # Sort by date string (ISO format sorts correctly)
    dated_entities.sort(key=lambda x: x[1])

    result.result_table = [
        {"entity": name, field: date} for name, date in dated_entities
    ]
    result.included_entities = [name for name, _ in dated_entities]
    result.missing_field_entities = missing

    if dated_entities:
        result.result_list = [dated_entities[-1][0]]  # Latest
        result.result_value = dated_entities[-1][1]
        result.computation_detail = (
            f"Sorted {len(dated_entities)} entities by '{field}', "
            f"latest: '{dated_entities[-1][0]}' ({dated_entities[-1][1]})"
        )

    return result


def _reduce_synthesis(plan: QueryPlan, ledger: EvidenceLedger) -> OperationResult:
    """Collect evidence for multi-hop or general synthesis.

    The actual synthesis is done by the answer model using
    only the evidence in this result.
    """
    result = OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        included_entities=[e.entity_name for e in ledger.entities],
    )

    # Build a table of all entity fields for the answer model
    table: list[dict[str, Any]] = []
    for entity in ledger.entities:
        row: dict[str, Any] = {"entity": entity.entity_name}
        row.update(entity.fields)
        table.append(row)

    result.result_table = table
    result.computation_detail = (
        f"Collected {len(ledger.items)} evidence items from "
        f"{len(ledger.entities)} entities for synthesis"
    )
    return result
