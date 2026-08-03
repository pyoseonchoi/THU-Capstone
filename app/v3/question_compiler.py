"""Deterministic question decomposition and adaptive strategy routing."""

from __future__ import annotations

import re

from app.schemas import QuestionRequest
from app.v3.models import Strategy, V3QuestionPlan


def _normalize_category(category: str) -> str:
    return category.strip().casefold().replace("-", "_").replace(" ", "_")


def _candidate_topics(question: str) -> list[str]:
    text = question.strip()
    if text.count("—") >= 2:
        body = text.split("—", 2)[1]
    else:
        match = re.search(
            r"\.\s*Of\s+(.+?),\s*which\s+(?:one|is)",
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


def _structured_metadata(question: str) -> dict[str, object]:
    folded = question.casefold()
    target_field = ""
    for terms, field in (
        (("highest point", "highest summit"), "highest_point"),
        (("area", "covers"), "area"),
        (("visitor",), "annual_visitors"),
        (("oldest tree", "estimated age"), "oldest_tree_age"),
        (("eruption",), "first_recorded_eruption_year"),
    ):
        if any(term in folded for term in terms):
            target_field = field
            break
    threshold_match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:metres?|meters?|m)\b", folded)
    threshold = (
        float(threshold_match.group(1).replace(",", ""))
        if threshold_match
        else None
    )
    operation = ""
    if any(term in folded for term in ("largest", "highest", "oldest", "most")):
        operation = "argmax"
    elif "how many" in folded:
        operation = "count"
    return {
        "target_field": target_field,
        "threshold": threshold,
        "operation": operation,
    }


def compile_question(question: QuestionRequest) -> V3QuestionPlan:
    """Compile a question without relying on a fragile free-form LLM operator."""
    category = _normalize_category(question.category)
    folded = question.question.casefold()

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
    elif any(term in folded for term in ("how many", "largest", "highest", "oldest")):
        strategy = Strategy.STRUCTURED_REDUCE
    else:
        strategy = Strategy.HIERARCHICAL_SYNTHESIS

    slots: list[str] = []
    if "how many" in folded:
        slots.append("count")
    if "name" in folded or "which" in folded:
        slots.append("entities")
    if any(term in folded for term in ("year", "when", "date")):
        slots.append("date")
    if strategy == Strategy.HIERARCHICAL_SYNTHESIS:
        slots.extend(["thesis", "representative_examples"])
    return V3QuestionPlan(
        question_id=question.question_id,
        question=question.question,
        category=category,
        strategy=strategy,
        candidate_topics=_candidate_topics(question.question),
        required_slots=list(dict.fromkeys(slots)),
        metadata=_structured_metadata(question.question),
    )


def compile_questions(questions: list[QuestionRequest]) -> list[V3QuestionPlan]:
    return [compile_question(question) for question in questions]
