"""Deterministic reducers for V3 exhaustive map results."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.field_names import canonical_field_name
from app.reduction.normalizer import parse_number
from app.v3.models import (
    CompiledDocument,
    EvidenceCandidate,
    EvidencePacket,
    ExecutionResult,
    Strategy,
    V3MapResult,
    V3QuestionPlan,
)

TERMINAL_STATUSES = {"evidence_found", "no_evidence"}


@dataclass(frozen=True)
class _Row:
    """One record's value for the field a structured question targets."""

    entity: str
    value: float
    unit: str
    page: int
    quote: str
    # What the figure is measured on, and the group the record belongs to.
    # A question asking which record leads usually also asks what the leading
    # thing is called and where it is.
    subject: str = ""
    group: str = ""


def _dominant_unit(rows) -> str:
    """Return the unit the parsed rows agree on, if they agree at all."""
    units = Counter(row.unit.casefold() for row in rows if row.unit)
    if not units:
        return ""
    unit, count = units.most_common(1)[0]
    return unit if count >= max(2, sum(units.values()) * 0.6) else ""


COVERED_STATUSES = {*TERMINAL_STATUSES, "uncertain"}
# Words that carry no topic: interrogatives, and the words a question uses to
# address the document itself rather than its subject.
_CONCEPT_STOPWORDS = frozenset({
    "about", "across", "all", "also", "among", "another", "any", "been",
    "between", "both", "each", "either", "entries", "entry", "every",
    "examples", "explain", "from", "give", "handle", "have", "here", "how",
    "identify", "into", "its", "itself", "list", "many", "more", "most",
    "much", "name", "over", "overall", "same", "several", "some", "such",
    "taking", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "using", "well", "were",
    "what", "when", "where", "which", "while", "whole", "with", "within",
})


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _concept_terms(question: str) -> set[str]:
    folded = question.casefold()
    terms = {
        term
        for term in re.findall(r"[a-z][a-z-]{3,}", folded)
        if term not in _CONCEPT_STOPWORDS
    }
    return terms


def _is_relevant_synthesis_evidence(
    plan: V3QuestionPlan,
    candidate: EvidenceCandidate,
) -> bool:
    if plan.strategy != Strategy.HIERARCHICAL_SYNTHESIS:
        return True
    terms = _concept_terms(plan.question)
    if not terms:
        return True
    haystack = _normalized_text(
        " ".join((candidate.claim, candidate.exact_quote, candidate.field, candidate.role))
    )
    return any(term in haystack for term in terms)


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
    dropped_irrelevant = 0
    for result in relevant:
        for item in result.evidence:
            if not _is_relevant_synthesis_evidence(plan, item):
                dropped_irrelevant += 1
                continue
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
    if dropped_irrelevant:
        warnings.append(
            f"Dropped {dropped_irrelevant} off-topic synthesis evidence items"
        )

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


def _topic_example(
    assessments: list,
    document: CompiledDocument | None,
) -> str:
    """Name where a present topic is discussed, preferring substantive cover.

    Establishing that the other options are present is half of what an absence
    question asks, and naming the record that carries each one is the evidence
    for that half; without it the claim is unsupported assertion.
    """
    if document is None:
        return ""
    titles = {record.record_id: record.title for record in document.records}
    ranked = sorted(
        (item for item in assessments if item.exact_quote),
        key=lambda item: (item.level != "substantive", item.page),
    )
    for item in ranked:
        title = titles.get(item.record_id, "")
        if title and not title.startswith("Record "):
            return title
    return ""


def reduce_absence(
    plan: V3QuestionPlan,
    results: list[V3MapResult],
    expected_record_ids: set[str],
    *,
    allow_partial: bool = False,
    document: CompiledDocument | None = None,
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
        return None

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
    covered: list[str] = []
    for topic in present_topics:
        example = _topic_example(by_topic[topic], document)
        covered.append(f"{topic} (for example at {example})" if example else topic)
    answer = (
        f"{missing_topic} is the only subject never {qualifier} anywhere in the book. "
        f"The other subjects are covered: {', '.join(covered)}."
    )
    evidence = [
        f"{topic}: {assessment.exact_quote} (page {assessment.page})"
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
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=evidence,
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
    document: CompiledDocument | None = None,
) -> ExecutionResult | None:
    """Apply count/argmax over compiled facts, topped up by mapped evidence.

    The compiler parses fact cards deterministically and quotes what it binds,
    so where it bound a value that value is exact. The mapper re-reads prose
    with a small model and misses rows that plainly state the figure, so using
    it to re-derive what was already parsed throws away the reliable half of
    the evidence. Compiled facts are therefore authoritative per record, and
    mapped evidence only covers records the compiler left empty.
    """
    if plan.strategy != Strategy.STRUCTURED_REDUCE or not packet.complete:
        return None
    target = str(plan.metadata.get("target_field", ""))
    operation = str(plan.metadata.get("operation", ""))
    threshold = plan.metadata.get("threshold")
    if not target or operation not in {"count", "argmax"}:
        return None

    rows: dict[str, _Row] = {}
    if document is not None:
        for record in document.records:
            fact = next(
                (item for item in record.number_facts if item.field == target),
                None,
            )
            if fact is not None:
                rows[record.record_id] = _Row(
                    entity=record.title or record.record_id,
                    value=fact.value,
                    unit=fact.unit,
                    page=fact.page,
                    quote=fact.quote,
                    subject=fact.subject,
                    group=record.country,
                )
    expected_unit = _dominant_unit(rows.values())
    for candidate in packet.evidence:
        if candidate.record_id in rows:
            continue
        if canonical_field_name(candidate.field) != target:
            continue
        value = _candidate_number(candidate)
        if value is None:
            continue
        # A figure stated in a different unit than the parsed rows use is a
        # misread of some other number on the page, and one such value is
        # enough to take over an extremum. Saying no unit at all contradicts
        # nothing, and rejecting those would discard every row the mapper
        # recovered for a record the parser missed.
        stated = candidate.unit.casefold().strip()
        if expected_unit and stated and stated != expected_unit:
            continue
        rows[candidate.record_id] = _Row(
            entity=candidate.entity or candidate.record_id,
            value=value,
            unit=candidate.unit,
            page=candidate.page,
            quote=candidate.exact_quote,
        )
    if not rows:
        return None

    if operation == "count":
        selected = [
            row
            for row in rows.values()
            if threshold is None or row.value >= float(threshold)
        ]
        entities = list(dict.fromkeys(row.entity for row in selected))
        answer = f"{len(entities)} entities qualify: {', '.join(entities)}."
        used = selected
    else:
        ranked = sorted(rows.values(), key=lambda row: row.value, reverse=True)
        best = ranked[0]
        unit = f" {best.unit}" if best.unit else ""
        named = f" ({best.subject})" if best.subject else ""
        where = f", in {best.group}" if best.group else ""
        answer = (
            f"{best.entity} reports the highest value: "
            f"{best.value:g}{unit}{named}{where}."
        )
        if len(ranked) > 1:
            comparison = ", ".join(
                f"{row.entity} at {row.value:g}{f' {row.unit}' if row.unit else ''}"
                for row in ranked[1:3]
            )
            answer += f" The next highest are {comparison}."
        used = ranked[:3]

    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[row.quote for row in used],
        source_pages=sorted({row.page for row in used}),
        complete=True,
        strategy=plan.strategy,
    )


def status_counts(
    question_id: str,
    results: list[V3MapResult],
) -> Counter[str]:
    return Counter(result.status for result in results if result.question_id == question_id)
