"""Deterministic reducers for V3 exhaustive map results."""

from __future__ import annotations

import re
from collections import Counter

from app.field_names import canonical_field_name
from app.reduction.normalizer import parse_number
from app.v3.models import (
    EvidenceCandidate,
    EvidencePacket,
    ExecutionResult,
    Strategy,
    V3MapResult,
    V3QuestionPlan,
)

TERMINAL_STATUSES = {"evidence_found", "no_evidence"}
COVERED_STATUSES = {*TERMINAL_STATUSES, "uncertain"}


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def build_evidence_packet(
    plan: V3QuestionPlan,
    results: list[V3MapResult],
    expected_record_ids: set[str],
) -> EvidencePacket:
    """Reduce all record outputs without relevance ranking or top-k selection."""
    relevant = [result for result in results if result.question_id == plan.question_id]
    by_record = {result.record_id: result for result in relevant}
    warnings: list[str] = []
    missing = expected_record_ids - set(by_record)
    nonterminal = [result.record_id for result in relevant if result.status not in COVERED_STATUSES]
    uncertain = [result.record_id for result in relevant if result.status == "uncertain"]
    if missing:
        warnings.append(f"Missing {len(missing)} record mappings")
    if nonterminal:
        warnings.append(f"{len(nonterminal)} record mappings failed technically")
    if uncertain:
        warnings.append(f"{len(uncertain)} record mappings are explicitly uncertain")

    evidence: list[EvidenceCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for result in relevant:
        for item in result.evidence:
            key = (
                item.record_id,
                _normalized_text(item.exact_quote),
                _normalized_text(item.claim),
            )
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    evidence.sort(key=lambda item: (item.page, item.record_id, item.claim))

    topic_summary: dict[str, dict[str, int]] = {}
    for result in relevant:
        for assessment in result.topics:
            counts = topic_summary.setdefault(assessment.topic, {})
            counts[assessment.level] = counts.get(assessment.level, 0) + 1

    return EvidencePacket(
        question_id=plan.question_id,
        records_scanned=len(by_record),
        expected_records=len(expected_record_ids),
        evidence=evidence,
        topic_summary=topic_summary,
        complete=(not missing and not nonterminal and set(by_record) == expected_record_ids),
        warnings=warnings,
    )


def reduce_absence(
    plan: V3QuestionPlan,
    results: list[V3MapResult],
    expected_record_ids: set[str],
    *,
    allow_partial: bool = False,
) -> ExecutionResult | None:
    """Reduce a multiple-choice absence matrix, optionally as best effort."""
    if not plan.candidate_topics:
        return None
    relevant = [result for result in results if result.question_id == plan.question_id]
    complete_coverage = {result.record_id for result in relevant} == expected_record_ids and all(
        result.status in TERMINAL_STATUSES for result in relevant
    )
    if not complete_coverage and not allow_partial:
        return None

    by_topic: dict[str, list] = {topic: [] for topic in plan.candidate_topics}
    for result in relevant:
        assessment_map = {_normalized_text(item.topic): item for item in result.topics}
        for topic in plan.candidate_topics:
            assessment = assessment_map.get(_normalized_text(topic))
            if assessment is not None and assessment.level != "uncertain":
                by_topic[topic].append(assessment)

    if complete_coverage and any(
        len(assessments) != len(expected_record_ids) for assessments in by_topic.values()
    ):
        return None

    substantive_only = "never substantively" in plan.question.casefold()
    support_counts: dict[str, int] = {}
    for topic, assessments in by_topic.items():
        disqualifying = (
            {"substantive"}
            if substantive_only
            else {
                "substantive",
                "mention_only",
            }
        )
        support_counts[topic] = sum(item.level in disqualifying for item in assessments)
    absent = [topic for topic, count in support_counts.items() if count == 0]
    if len(absent) != 1:
        if not allow_partial or not support_counts:
            return None
        ordered = sorted(support_counts.items(), key=lambda item: item[1])
        unique_low_outlier = (
            len(ordered) > 1
            and ordered[0][1] < ordered[1][1]
            and ordered[1][1] >= max(2, ordered[0][1] + 2)
        )
        if not unique_low_outlier:
            return None
        absent = [ordered[0][0]]

    if allow_partial and not complete_coverage:
        present = set(plan.candidate_topics) - set(absent)
        if any(
            not any(
                item.level
                in (
                    {"substantive"}
                    if substantive_only
                    else {
                        "substantive",
                        "mention_only",
                    }
                )
                for item in by_topic[topic]
            )
            for topic in present
        ):
            return None

    missing_topic = absent[0]
    present_topics = [topic for topic in plan.candidate_topics if topic != missing_topic]
    qualifier = "substantively discussed" if substantive_only else "mentioned or raised"
    answer = (
        f"{missing_topic} is the only subject never {qualifier} anywhere in the book. "
        f"The other subjects are covered: {', '.join(present_topics)}."
    )
    evidence = [
        f"{topic}: {assessment.exact_quote} (page {assessment.page})"
        for topic in present_topics
        for assessment in by_topic[topic]
        if assessment.level in {"substantive", "mention_only"} and assessment.exact_quote
    ]
    evidence_pages = [
        assessment.page
        for topic in present_topics
        for assessment in by_topic[topic]
        if assessment.level in {"substantive", "mention_only"} and assessment.exact_quote
    ]
    pages = sorted(
        {
            assessment.page
            for topic in present_topics
            for assessment in by_topic[topic]
            if assessment.page
        }
    )
    warnings: list[str] = []
    if not complete_coverage:
        terminal_records = {
            result.record_id for result in relevant if result.status in TERMINAL_STATUSES
        }
        warnings.append(
            "Best-effort absence conclusion after targeted repair: "
            f"{len(terminal_records)}/{len(expected_record_ids)} records terminal"
        )
    if support_counts[missing_topic] > 0:
        warnings.append("Ignored a unique low-support semantic outlier for the absent topic")
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=evidence,
        evidence_pages=evidence_pages,
        source_pages=pages,
        complete=True,
        strategy=plan.strategy,
        warnings=warnings,
    )


def _candidate_number(candidate: EvidenceCandidate) -> float | None:
    if isinstance(candidate.value, (int, float)):
        return float(candidate.value)
    if candidate.value is not None:
        parsed = parse_number(str(candidate.value))
        if parsed is not None:
            return parsed
    return parse_number(candidate.exact_quote)


def reduce_mapped_structure(
    plan: V3QuestionPlan,
    packet: EvidencePacket,
) -> ExecutionResult | None:
    """Apply count/argmax in Python when compile-time facts were unavailable."""
    if plan.strategy != Strategy.STRUCTURED_REDUCE or not packet.complete:
        return None
    target = str(plan.metadata.get("target_field", ""))
    operation = str(plan.metadata.get("operation", ""))
    threshold = plan.metadata.get("threshold")
    if not target or operation not in {"count", "argmax"}:
        return None

    candidates: list[tuple[EvidenceCandidate, float]] = []
    for candidate in packet.evidence:
        if canonical_field_name(candidate.field) != target:
            continue
        value = _candidate_number(candidate)
        if value is not None:
            candidates.append((candidate, value))
    if not candidates:
        return None

    if operation == "count":
        selected = [
            (candidate, value)
            for candidate, value in candidates
            if threshold is None or value >= float(threshold)
        ]
        entities = list(
            dict.fromkeys(candidate.entity or candidate.record_id for candidate, _ in selected)
        )
        answer = f"{len(entities)} entities qualify: {', '.join(entities)}."
        used = [candidate for candidate, _ in selected]
    else:
        candidate, value = max(candidates, key=lambda item: item[1])
        entity = candidate.entity or candidate.record_id
        unit = f" {candidate.unit}" if candidate.unit else ""
        answer = f"{entity} has the maximum reported value: {value:g}{unit}."
        used = [candidate]

    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[item.exact_quote for item in used],
        evidence_pages=[item.page for item in used],
        source_pages=sorted({item.page for item in used}),
        complete=True,
        strategy=plan.strategy,
    )


def status_counts(
    question_id: str,
    results: list[V3MapResult],
) -> Counter[str]:
    return Counter(result.status for result in results if result.question_id == question_id)
