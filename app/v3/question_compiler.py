"""Deterministic question decomposition and composable operation routing."""

from __future__ import annotations

import re

from app.schemas import QuestionRequest
from app.v3.models import (
    OperationKind,
    OperationStep,
    Strategy,
    V3QuestionPlan,
)

_FIELD_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("highest operating point", "operating elevation"), "highest_point"),
    (("highest point", "highest summit", "highest peak"), "highest_point"),
    ((
        "annual visiting researchers",
        "annual number of visiting researchers",
        "visiting researchers per year",
    ), "annual_visitors"),
    (("annual visitors", "visitor figure", "number of visitors"), "annual_visitors"),
    (("area monitored", "monitored area", "largest area", "covers"), "area"),
    (("year established", "establishment year", "station's founding"), "establishment_year"),
    (("estimated age", "oldest tree"), "oldest_tree_age"),
    (("first recorded eruption",), "first_recorded_eruption_year"),
)

_GENERIC_CAPITALIZED = {
    "Across",
    "Among",
    "Counting",
    "Explain",
    "For",
    "How",
    "Identify",
    "In",
    "Of",
    "State",
    "Taking",
    "The",
    "These",
    "Two",
    "What",
    "Which",
}


def _normalize_category(category: str) -> str:
    return category.strip().casefold().replace("-", "_").replace(" ", "_")


def _candidate_topics(question: str) -> list[str]:
    """Extract the explicitly enumerated alternatives in an absence question."""
    text = question.strip()
    dash_match = re.search(
        r"\s\u2014\s(.+?)\s\u2014\s+which\b",
        text,
        re.IGNORECASE,
    )
    if dash_match:
        body = dash_match.group(1)
    else:
        match = re.search(
            r"(?:Of|of)\s+(.+?),\s+which\s+(?:one\s+)?(?:is|was|has)",
            text,
            re.IGNORECASE,
        )
        body = match.group(1) if match else ""
    if not body:
        return []
    parts = [part.strip(" .") for part in re.split(r",\s*", body) if part.strip()]
    if parts and parts[-1].casefold().startswith("and "):
        parts[-1] = parts[-1][4:].strip()
    return parts


def _target_field(question: str) -> str:
    folded = question.casefold()
    for terms, field in _FIELD_ALIASES:
        if any(term in folded for term in terms):
            return field
    return ""


def _threshold(question: str) -> float | None:
    match = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:metres?|meters?|m|km|sq\s*(?:km|miles?))\b",
        question,
        re.IGNORECASE,
    )
    return float(match.group(1).replace(",", "")) if match else None


def _status_values(question: str) -> list[str]:
    folded = question.casefold()
    return [
        status
        for status in ("inscribed", "nominated", "tentative")
        if status in folded
    ]


def _operation_steps(question: str, category: str, target: str) -> list[OperationStep]:
    folded = question.casefold()
    threshold = _threshold(question)
    steps: list[OperationStep] = []

    statuses = _status_values(question)
    if statuses and "among" in folded and target:
        steps.append(OperationStep(
            kind=OperationKind.FILTER,
            field="designation_status",
            comparator="in",
            values=statuses,
        ))

    has_superlative = any(
        term in folded for term in ("largest", "highest", "oldest", "most")
    )
    is_count = folded.startswith("counting ") or (
        "how many" in folded
        and "by how many" not in folded
        and not has_superlative
    )
    if is_count:
        if target and threshold is not None:
            comparator = "gte"
            if any(term in folded for term in ("above", "over", "more than")):
                comparator = "gt"
            steps.append(OperationStep(
                kind=OperationKind.FILTER,
                field=target,
                comparator=comparator,
                value=threshold,
            ))
        steps.append(OperationStep(kind=OperationKind.COUNT, field=target))
        if "name" in folded or "list" in folded:
            steps.append(OperationStep(kind=OperationKind.LIST, field=target))
    elif any(term in folded for term in ("largest", "highest", "oldest", "most")) and target:
        steps.append(OperationStep(kind=OperationKind.ARGMAX, field=target))

    if (
        category == "cross_section"
        and "event" in folded
        and any(term in folded for term in ("closer", "difference", "by how many years"))
    ):
        steps.extend([
            OperationStep(
                kind=OperationKind.JOIN,
                field="event_year",
                group_by="entity",
            ),
            OperationStep(
                kind=OperationKind.DATE_DIFFERENCE,
                field="establishment_year",
                group_by="entity",
            ),
            OperationStep(kind=OperationKind.COMPARE, field="year_difference"),
        ])
    elif category == "cross_section" and any(
        term in folded
        for term in ("straddles a border", "straddle a border", "transboundary", "paired")
    ):
        steps.append(OperationStep(
            kind=OperationKind.JOIN,
            field="cross_border_relation",
            group_by="entity",
        ))
    elif category == "contradiction":
        steps.append(OperationStep(kind=OperationKind.COMPARE, field=target))

    return steps


def _entity_hints(question: str) -> list[str]:
    hints: list[str] = []
    pattern = r"\b[A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,4}"
    for match in re.finditer(pattern, question):
        text = match.group(0).strip()
        if text in _GENERIC_CAPITALIZED or len(text) < 3:
            continue
        hints.append(text.removesuffix("'s"))
    return list(dict.fromkeys(hints))


def _structured_metadata(
    question: str,
    operations: list[OperationStep],
    target: str,
) -> dict[str, object]:
    terminal = next(
        (
            step.kind.value
            for step in reversed(operations)
            if step.kind in {OperationKind.COUNT, OperationKind.ARGMAX, OperationKind.ARGMIN}
        ),
        "",
    )
    filter_step = next(
        (step for step in operations if step.kind == OperationKind.FILTER),
        None,
    )
    return {
        "target_field": target,
        "threshold": _threshold(question),
        "operation": terminal,
        "filter_field": filter_step.field if filter_step else "",
        "filter_values": filter_step.values if filter_step else [],
    }


def compile_question(question: QuestionRequest) -> V3QuestionPlan:
    """Compile a question into a validated strategy and operation sequence."""
    category = _normalize_category(question.category)
    folded = question.question.casefold()
    target = _target_field(question.question)
    operations = _operation_steps(question.question, category, target)

    if category == "absence" or any(
        term in folded for term in ("never mentioned", "never raised", "never substantively")
    ):
        strategy = Strategy.ABSENCE_MATRIX
    elif category == "contradiction" or any(
        term in folded for term in ("contradict", "inconsisten", "different units")
    ):
        strategy = Strategy.CLAIM_COMPARE
    elif category in {"global_synthesis", "cross_section"}:
        strategy = Strategy.HIERARCHICAL_SYNTHESIS
    elif category in {"aggregation", "superlative"}:
        strategy = Strategy.STRUCTURED_REDUCE
    elif category in {"needle", "needle_(control)", "needle_control"}:
        strategy = Strategy.EXHAUSTIVE_LOOKUP
    elif operations:
        strategy = Strategy.STRUCTURED_REDUCE
    else:
        strategy = Strategy.HIERARCHICAL_SYNTHESIS

    slots: list[str] = []
    if any(step.kind == OperationKind.COUNT for step in operations):
        slots.append("count")
    if "name" in folded or "which" in folded or "identify" in folded:
        slots.append("entities")
    if any(term in folded for term in ("year", "when", "date")):
        slots.append("date")
    if strategy == Strategy.CLAIM_COMPARE:
        slots.extend(["claim", "counterevidence"])
    if strategy == Strategy.HIERARCHICAL_SYNTHESIS:
        slots.extend(["thesis", "representative_examples"])

    return V3QuestionPlan(
        question_id=question.question_id,
        question=question.question,
        category=category,
        strategy=strategy,
        candidate_topics=_candidate_topics(question.question),
        required_slots=list(dict.fromkeys(slots)),
        operations=operations,
        target_fields=[target] if target else [],
        entity_hints=_entity_hints(question.question),
        metadata=_structured_metadata(question.question, operations, target),
    )


def compile_questions(questions: list[QuestionRequest]) -> list[V3QuestionPlan]:
    return [compile_question(question) for question in questions]
