"""Deterministic question decomposition and composable operation routing."""

from __future__ import annotations

import re

from app.schemas import QuestionRequest
from app.v3.models import (
    CompiledDocument,
    OperationKind,
    OperationStep,
    Strategy,
    V3QuestionPlan,
)

_FIELD_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("highest operating point", "operating elevation", "highest crest"), "highest_point"),
    (("highest point", "highest summit", "highest peak"), "highest_point"),
    ((
        "annual visiting researchers",
        "annual number of visiting researchers",
        "visiting researchers per year",
    ), "annual_visitors"),
    (("annual visitors", "visitor figure", "number of visitors"), "annual_visitors"),
    ((
        "annual generation",
        "annual output",
        "pumping-equivalent output",
        "gigawatt-hours",
    ), "annual_output"),
    ((
        "area monitored",
        "monitored area",
        "monitored reservoir",
        "project area",
        "largest area",
        "covers",
    ), "area"),
    ((
        "year established",
        "establishment year",
        "station's founding",
        "commissioned",
        "commissioning year",
    ), "establishment_year"),
    (("life expectancy",), "life_expectancy_2023"),
    (("gross national income per capita", "gni per capita"), "gni_per_capita_2023"),
    (("human development index value", "hdi value"), "hdi_2023"),
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
        match = re.search(r"\bof\s+(.+?),\s+which\b", text, re.IGNORECASE)
        body = match.group(1) if match else ""
    if not body:
        colon_match = re.search(r":\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if colon_match:
            body = colon_match.group(1).strip().rstrip("?. ")
    if not body:
        return []
    parts = [part.strip(" .") for part in re.split(r",\s*", body) if part.strip()]
    if parts and re.match(r"^(?:and|or)\s+", parts[-1], re.I):
        parts[-1] = re.sub(r"^(?:and|or)\s+", "", parts[-1], flags=re.I).strip()
    elif len(parts) >= 3 and re.search(r"\s+(?:and|or)\s+", parts[-1], re.I):
        left, right = re.split(r"\s+(?:and|or)\s+", parts[-1], maxsplit=1, flags=re.I)
        parts[-1:] = [left.strip(), right.strip()]
    elif len(parts) == 1 and re.search(r"\s+(?:and|or)\s+", parts[0], re.I):
        parts = [part.strip() for part in re.split(r"\s+(?:and|or)\s+", parts[0], flags=re.I)]
    return parts


def _target_field(
    question: str,
    document: CompiledDocument | None = None,
) -> str:
    folded = question.casefold()
    for terms, field in _FIELD_ALIASES:
        if any(term in folded for term in terms):
            return field
    if document is not None:
        scored: list[tuple[int, str]] = []
        question_terms = set(re.findall(r"[a-z0-9]+", folded))
        for field in document.field_catalog:
            terms = set(field.casefold().split("_")) - {"2023", "year"}
            overlap = len(terms & question_terms)
            if overlap:
                scored.append((overlap, field))
        if scored:
            return max(scored, key=lambda item: (item[0], len(item[1])))[1]
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


def _mentioned_countries(
    question: str,
    document: CompiledDocument | None,
) -> list[str]:
    if document is None:
        return []
    folded = question.casefold()
    countries = list(dict.fromkeys(record.country for record in document.records if record.country))
    return [country for country in countries if country.casefold() in folded]


def _operation_steps(
    question: str,
    category: str,
    target: str,
    document: CompiledDocument | None = None,
) -> list[OperationStep]:
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

    if "ranked" in folded and target:
        steps.append(OperationStep(
            kind=OperationKind.FILTER,
            field="rank",
            comparator="not_null",
        ))

    countries = _mentioned_countries(question, document)
    if len(countries) == 1 and not (
        "in total" in folded and "how many" in folded
    ):
        steps.append(OperationStep(
            kind=OperationKind.FILTER,
            field="country",
            comparator="eq",
            value=countries[0],
        ))

    has_superlative = any(
        term in folded
        for term in (
            "largest",
            "highest",
            "oldest",
            "most",
            "greatest",
            "maximum",
            "lowest",
            "earliest",
            "latest",
        )
    )
    entity_count_nouns = (
        "how many projects",
        "how many stations",
        "how many parks",
        "how many countries",
        "how many entries",
        "how many items",
        "how many figures",
    )
    is_count = folded.startswith("counting ") or any(
        phrase in folded for phrase in entity_count_nouns
    ) or (
        "how many" in folded
        and "by how many" not in folded
        and not has_superlative
        and category == "aggregation"
    )
    if is_count and "each" in folded and "group" in folded:
        steps.append(OperationStep(
            kind=OperationKind.GROUP_BY,
            field="group",
            group_by="group",
        ))
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
    elif has_superlative and target:
        minimum = any(term in folded for term in ("lowest", "earliest")) or (
            "oldest" in folded and target == "establishment_year"
        )
        steps.append(OperationStep(
            kind=OperationKind.ARGMIN if minimum else OperationKind.ARGMAX,
            field=target,
        ))

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


def _entity_hints(
    question: str,
    document: CompiledDocument | None = None,
) -> list[str]:
    if document is not None:
        folded = question.casefold()
        matched = [
            record.title
            for record in document.records
            if record.title.casefold() in folded
            or record.title.split()[0].casefold() in folded
        ]
        if matched:
            return list(dict.fromkeys(matched))
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
    table_match = re.search(r"\btable\s+(\d+)\b", question, re.I)
    table_identifiers = list(dict.fromkeys(
        match.group(1) for match in re.finditer(r"\btable\s+([\dA-Za-z.]+)", question, re.I)
    ))
    folded_question = question.casefold()
    mentions_contents_category = any(
        category in folded_question
        for category in ("figures", "tables", "boxes", "spotlights")
    )
    source_view = (
        "contents"
        if "according to the contents" in folded_question
        or (mentions_contents_category and "chapter" in folded_question)
        else ""
    )
    return {
        "target_field": target,
        "threshold": _threshold(question),
        "operation": terminal,
        "filter_field": filter_step.field if filter_step else "",
        "filter_values": filter_step.values if filter_step else [],
        "filters": [
            step.model_dump(mode="json")
            for step in operations
            if step.kind == OperationKind.FILTER
        ],
        "source_table_number": int(table_match.group(1)) if table_match else None,
        "source_table_identifiers": table_identifiers,
        "source_view": source_view,
    }


def compile_question(
    question: QuestionRequest,
    document: CompiledDocument | None = None,
) -> V3QuestionPlan:
    """Compile a question into a validated strategy and operation sequence."""
    category = _normalize_category(question.category)
    folded = question.question.casefold()
    target = _target_field(question.question, document)
    if not target and "ranked" in folded and "group" in folded:
        target = "rank"
    operations = _operation_steps(question.question, category, target, document)

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
        target_fields=(
            [target]
            if target and (strategy != Strategy.HIERARCHICAL_SYNTHESIS or operations)
            else []
        ),
        entity_hints=_entity_hints(question.question, document),
        metadata=_structured_metadata(question.question, operations, target),
    )


def compile_questions(
    questions: list[QuestionRequest],
    document: CompiledDocument | None = None,
) -> list[V3QuestionPlan]:
    return [compile_question(question, document) for question in questions]
