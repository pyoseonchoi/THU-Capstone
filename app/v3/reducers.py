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
_CONCEPT_STOPWORDS = {
    "across",
    "all",
    "book",
    "different",
    "does",
    "each",
    "entries",
    "examples",
    "explain",
    "guide",
    "handle",
    "identify",
    "overall",
    "profile",
    "profiles",
    "station",
    "stations",
    "using",
    "what",
    "which",
}
_CONCEPT_EXPANSIONS = {
    "fire": {"fire", "burn", "burning", "wildfire", "post-fire"},
    "wildlife": {
        "wildlife",
        "species",
        "population",
        "recovery",
        "reintroduction",
        "conservation",
        "threatened",
    },
    "livelihoods": {
        "livelihood",
        "families",
        "community",
        "communities",
        "herding",
        "cutting",
        "fishers",
        "traditional",
        "councils",
    },
    "glaciation": {"glacier", "glacial", "ice age", "ice cap", "moraine"},
    "border": {"border", "cross-border", "transboundary", "paired", "shared"},
}


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _concept_terms(question: str) -> set[str]:
    folded = question.casefold()
    terms = {
        term
        for term in re.findall(r"[a-z][a-z-]{3,}", folded)
        if term not in _CONCEPT_STOPWORDS
    }
    for trigger, expansions in _CONCEPT_EXPANSIONS.items():
        if trigger in folded or any(expansion in folded for expansion in expansions):
            terms.update(expansions)
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


def _topic_example(assessments: list, document) -> str:
    """Name where a present topic is discussed, preferring substantive cover.

    Establishing that the other candidate topics are present is half of what
    an absence question asks; naming the record that carries each one is the
    evidence for that half, without which the claim is an unsupported
    assertion. Reuses `document.records` (already loaded for every question on
    this document) rather than a new lookup structure.
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
    document=None,
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
    covered = []
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


def reduce_claim_compare(
    plan: V3QuestionPlan,
    packet: EvidencePacket,
) -> ExecutionResult | None:
    """Render a contradiction answer from typed values instead of free prose.

    The mapper already extracts a numeric `value` per evidence item and
    validates `exact_quote` as a literal source substring. Asking a model to
    then retype those numbers from memory while composing the final sentence
    reintroduces the transcription errors the typed extraction was meant to
    avoid -- so when both sides of the contradiction resolve to a real
    number, render the comparison directly from those values and skip free
    narration. Falls through to the existing synthesis path (returns None)
    whenever the evidence doesn't cleanly resolve to two typed numbers, so it
    only ever adds coverage rather than replacing it.
    """
    if plan.strategy != Strategy.CLAIM_COMPARE:
        return None
    numbered = [
        (item, _candidate_number(item))
        for item in packet.evidence
        if item.role in ("claim", "counterevidence")
    ]
    numbered = [(item, value) for item, value in numbered if value is not None]
    claims = [item for item, value in numbered if item.role == "claim"]
    counters = [(item, value) for item, value in numbered if item.role == "counterevidence"]
    if not claims or not counters:
        return None

    claim_item = max(claims, key=lambda item: item.confidence)
    claim_value = _candidate_number(claim_item)
    if claim_value is None:
        return None

    # A counterevidence item whose value equals the claim's isn't actually
    # conflicting -- two sources stating the same number aren't a
    # contradiction just because they're phrased differently. Prefer the
    # highest-confidence counter that genuinely differs, and fall through to
    # free-form synthesis (which can read the surrounding prose) if none do.
    differing = [
        (item, value) for item, value in counters if value != claim_value
    ]
    if not differing:
        return None
    counter_item, counter_value = max(differing, key=lambda pair: pair[0].confidence)

    def describe(item: EvidenceCandidate, value: float) -> str:
        has_unit = item.unit and item.unit.strip().casefold() != "none"
        unit = f" {item.unit}" if has_unit else ""
        entity = item.entity or item.record_id
        formatted = f"{value:g}{unit}"
        quote = item.exact_quote.strip()
        if len(quote) > 200:
            quote = quote[:200].rsplit(" ", 1)[0] + "..."
        return f"{entity} is recorded at {formatted} (\"{quote}\")"

    answer = (
        f"{describe(claim_item, claim_value)}. However, "
        f"{describe(counter_item, counter_value)}. These figures conflict."
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[claim_item.exact_quote, counter_item.exact_quote],
        source_pages=sorted({claim_item.page, counter_item.page}),
        complete=True,
        strategy=plan.strategy,
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
    if not target or operation not in {"count", "argmax", "argmin"}:
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
        seeking_max = operation != "argmin"
        ranked = sorted(candidates, key=lambda item: item[1], reverse=seeking_max)
        candidate, value = ranked[0]
        entity = candidate.entity or candidate.record_id
        unit = f" {candidate.unit}" if candidate.unit else ""
        verb = "the highest" if seeking_max else "the lowest"
        answer = f"{entity} reports {verb} value: {value:g}{unit}."
        if len(ranked) > 1:
            runner_up_label = "next highest" if seeking_max else "next lowest"
            comparison = ", ".join(
                f"{item[0].entity or item[0].record_id} at {item[1]:g}"
                f"{f' {item[0].unit}' if item[0].unit else ''}"
                for item in ranked[1:3]
            )
            answer += f" The {runner_up_label} are {comparison}."
        used = [item[0] for item in ranked[:3]]

    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[item.exact_quote for item in used],
        source_pages=sorted({item.page for item in used}),
        complete=True,
        strategy=plan.strategy,
    )


def status_counts(
    question_id: str,
    results: list[V3MapResult],
) -> Counter[str]:
    return Counter(result.status for result in results if result.question_id == question_id)
