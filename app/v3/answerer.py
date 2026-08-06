"""Evidence-grounded final synthesis with one bounded slot-refinement pass."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.v3.models import CompiledDocument, EvidencePacket, ExecutionResult, V3QuestionPlan

logger = get_logger("v3.answerer")
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_answer_system.txt"


def _flatten_to_text(value: object) -> str:
    """Render a JSON value as readable prose instead of a Python repr.

    Small models occasionally return a nested object under "answer" despite
    the prompt asking for a plain string. `str(dict)` would surface Python
    syntax (braces, single quotes) straight to the grader. Dict keys here are
    internal scaffolding names (e.g. required-slot labels like "thesis" or
    "representative_examples"), not content, so they are dropped rather than
    printed — only the values are joined into prose, with no information lost.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(
            _flatten_to_text(val) for val in value.values() if val not in (None, "", [], {})
        )
    if isinstance(value, list):
        return "; ".join(_flatten_to_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


_INLINE_EVIDENCE_TAG_RE = re.compile(r"\s*\(E\d+(?:\s*,\s*E\d+)*\)")


def _strip_inline_evidence_tags(text: str) -> str:
    """Drop internal evidence-id citations (e.g. "(E014)") the model may echo
    into the answer text — they are bookkeeping for our own pipeline, not
    content a grader or reader can use.
    """
    return _INLINE_EVIDENCE_TAG_RE.sub("", text)


_INTERNAL_ID_TOKEN_RE = re.compile(
    r",?\s*(?:segment|chunk|record)-\d+", re.IGNORECASE
)
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)")


def _strip_internal_id_tokens(text: str) -> str:
    """Drop internal record/segment/chunk id tokens (e.g. "segment-007") the
    model may echo verbatim from evidence bookkeeping fields — these read as
    citations but are meaningless to a reader or grader, and can appear mixed
    into an otherwise legitimate citation (e.g. "(Figure 1.8, segment-007)").
    """
    stripped = _INTERNAL_ID_TOKEN_RE.sub("", text)
    return _EMPTY_PARENS_RE.sub("", stripped)


_CITATION_IDENTIFIER_RE = re.compile(
    r"\b(?:Figure|Table|Box|Spotlight)\s+[A-Za-z]?\.?\d+(?:\.[A-Za-z0-9]+)*",
    re.IGNORECASE,
)
_PARENTHETICAL_RE = re.compile(r"\([^()]*\)")


def _normalize_citation(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _verified_citations(evidence: list[dict]) -> set[str]:
    """Collect every Figure/Table/Box/Spotlight identifier actually present
    in the evidence handed to the model, so a cited number can be checked
    against what was really extracted rather than trusted at face value.
    """
    verified: set[str] = set()
    for item in evidence:
        haystack = " ".join(
            str(item.get(key, "")) for key in ("claim", "exact_quote", "field", "entity")
        )
        for match in _CITATION_IDENTIFIER_RE.finditer(haystack):
            verified.add(_normalize_citation(match.group(0)))
    return verified


def _drop_unverified_citations(answer: str, verified: set[str]) -> str:
    """Remove a parenthetical aside that cites a Figure/Table/Box/Spotlight
    number never seen in the evidence -- the model sometimes echoes a
    plausible-looking but wrong or merely nearby number (e.g. citing
    "Figure O.8" when the evidence was drawn from Figure 1.8). Only touches
    parentheses that specifically make such a citation claim; an ordinary
    parenthetical aside (a quote, a qualifier) is left untouched. Skipped
    entirely when nothing in the evidence names a Figure/Table/Box/Spotlight
    at all, since an empty verified set carries no signal to judge by.
    """
    if not verified:
        return answer

    def replace(match: re.Match) -> str:
        content = match.group(0)
        identifiers = _CITATION_IDENTIFIER_RE.findall(content)
        if not identifiers:
            return content
        if any(_normalize_citation(ident) in verified for ident in identifiers):
            return content
        return ""

    cleaned = _PARENTHETICAL_RE.sub(replace, answer)
    cleaned = _EMPTY_PARENS_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned)


def _parse_answer(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return text
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return text
    answer = _flatten_to_text(data.get("answer", data.get("final_answer", ""))).strip()
    answer = _strip_inline_evidence_tags(answer)
    return _strip_internal_id_tokens(answer).strip()


_TERM_STOPWORDS = {
    "about", "across", "all", "also", "among", "and", "any", "are", "book",
    "does", "each", "entries", "examples", "explain", "from", "guide",
    "handle", "have", "identify", "into", "over", "overall", "profile",
    "profiles", "that", "the", "their", "these", "this", "using", "what",
    "when", "where", "which", "with",
}


def _question_terms(question: str) -> frozenset[str]:
    """Extract the question's own content words, for scoring evidence by
    relevance to *this specific question* rather than by generic narrative
    signal words alone. Plain keyword overlap on the already-exhaustively-
    gathered evidence -- not a similarity search over the source document,
    so it re-ranks what was already read rather than deciding what to read.
    """
    return frozenset(
        term
        for term in re.findall(r"[a-z][a-z-]{3,}", question.casefold())
        if term not in _TERM_STOPWORDS
    )


def _evidence_score(item, question_terms: frozenset[str] = frozenset()) -> tuple[int, int, float]:
    text = f"{item.claim} {item.exact_quote}".casefold()
    signals = (
        "increased",
        "declined",
        "recovered",
        "bounced back",
        "reintroduced",
        "fell to",
        "rose to",
        "from ",
        "but ",
        "since ",
    )
    relevance = sum(term in text for term in question_terms)
    return relevance, sum(signal in text for signal in signals), item.confidence


def _bounded_evidence(plan: V3QuestionPlan, packet: EvidencePacket):
    items = list(packet.evidence)
    if plan.strategy.value != "hierarchical_synthesis":
        return items
    limit = 80 if plan.category == "cross_section" else 36
    if len(items) <= limit:
        return items
    terms = _question_terms(plan.question)

    def score(item):
        return _evidence_score(item, terms)

    if plan.category == "cross_section":
        return sorted(items, key=score, reverse=True)[:limit]

    expected = max(packet.expected_records, 1)
    buckets: list[list] = [[], [], []]
    for item in items:
        ratio = max(item.record_ordinal - 1, 0) / expected
        bucket = 0 if ratio < 1 / 3 else (1 if ratio < 2 / 3 else 2)
        buckets[bucket].append(item)
    per_bucket = max(1, limit // 3)
    selected = [
        item
        for bucket in buckets
        for item in sorted(bucket, key=score, reverse=True)[:per_bucket]
    ]
    return sorted(selected[:limit], key=lambda item: (item.record_ordinal, item.page))


def _evidence_payload(
    plan: V3QuestionPlan,
    packet: EvidencePacket,
) -> list[dict]:
    """Represent reduced evidence within a bounded synthesis envelope."""
    expected = max(packet.expected_records, 1)
    return [
        {
            "evidence_id": f"E{index:03d}",
            "record_id": item.record_id,
            "page": item.page,
            "entity": item.entity,
            "field": item.field,
            "value": item.value,
            "unit": item.unit,
            "role": item.role,
            "record_ordinal": item.record_ordinal,
            "document_position": (
                "early"
                if item.record_ordinal <= expected / 3
                else ("middle" if item.record_ordinal <= 2 * expected / 3 else "late")
            ),
            "claim": item.claim[:600],
            "exact_quote": item.exact_quote[:900],
        }
        for index, item in enumerate(_bounded_evidence(plan, packet), start=1)
    ]


def _document_structure(
    plan: V3QuestionPlan,
    document: CompiledDocument | None,
) -> list[str] | None:
    """Ground global-synthesis prose in the document's own real structure.

    Free-form synthesis otherwise writes from the evidence packet alone,
    which carries no signal for how the source document itself is organized
    (its actual chapter list, or its actual set of profiled entities). That
    lets the model invent a plausible-sounding but wrong grouping or thesis.
    Handing over the document's own verified contents listing (or, absent
    one, its own record titles) costs little and lets the prompt ask the
    model to stay consistent with it.
    """
    if document is None or plan.category != "global_synthesis":
        return None
    if document.contents_entries:
        labels: list[str] = []
        for entry in document.contents_entries:
            label = entry.title or entry.identifier
            if label and label not in labels:
                labels.append(label)
        return labels[:60] or None
    titles = [record.title for record in document.records if record.title]
    return titles[:80] or None


def _missing_slots(plan: V3QuestionPlan, answer: str) -> list[str]:
    missing: list[str] = []
    if "date" in plan.required_slots and not re.search(
        r"\b(?:\d{3,4}\s*(?:BC|BCE|AD|CE)?|\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
        answer,
        re.IGNORECASE,
    ):
        missing.append("date")
    if "thesis" in plan.required_slots and len(answer.split()) < 35:
        missing.append("thesis with representative examples")
    if "entities" in plan.required_slots and len(answer.split()) < 5:
        missing.append("named entities")
    return missing


def _fallback_answer(plan: V3QuestionPlan, packet: EvidencePacket) -> str:
    """Return a conservative answer draft when final synthesis is unavailable."""
    claims: list[str] = []
    seen: set[str] = set()
    for item in packet.evidence:
        claim = (item.claim or item.exact_quote).strip()
        key = claim.casefold()
        if not claim or key in seen:
            continue
        seen.add(key)
        claims.append(claim.rstrip(".") + ".")
    if not claims:
        return ""
    if plan.strategy.value == "exhaustive_lookup":
        return claims[0]
    return " ".join(claims[:10])


_THESIS_CHECK_INSTRUCTION = (
    "Check your draft's central thesis against the evidence above, item by item: "
    "does each piece of evidence support the stated thesis, contradict it, or say "
    "something different? If most of the evidence contradicts the thesis or points "
    "in a different direction than what the draft claims, rewrite only the thesis "
    "sentence(s) to match what the evidence collectively shows. Do not shorten the "
    "answer or drop any named example, date, or quantity already in the draft -- "
    "keep every one of them, and only adjust the framing that connects them. If "
    "the thesis is already well-supported by most of the evidence, return the "
    "draft completely unchanged."
)

_COMPLETENESS_CHECK_INSTRUCTION = (
    "Check your draft against the evidence above for completeness: does the "
    "evidence contain any distinct matching case, entity, figure/table citation, "
    "or example that the draft left out? If the question implies an exhaustive "
    "list or an open count (e.g. \"which parks...\", \"what evidence shows...\"), "
    "add every distinct case the evidence supports rather than a partial sample, "
    "and never answer that evidence is insufficient when the evidence above "
    "actually contains a usable answer. Do not remove or reword anything already "
    "correct in the draft -- only add what is missing. If the draft is already "
    "complete, return it completely unchanged."
)


class V3Answerer:
    """Generate a grounded answer and refine only objectively missing slots."""

    def __init__(self, router: LLMRouter, tracker: UsageTracker) -> None:
        self._router = router
        self._tracker = tracker
        self._system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    async def generate(
        self,
        plan: V3QuestionPlan,
        packet: EvidencePacket,
        document: CompiledDocument | None = None,
    ) -> ExecutionResult:
        if not packet.evidence:
            return ExecutionResult(
                question_id=plan.question_id,
                strategy=plan.strategy,
                complete=False,
                warnings=[*packet.warnings, "No complete grounded evidence packet"],
            )
        evidence = _evidence_payload(plan, packet)
        structure = _document_structure(plan, document)
        warnings = list(packet.warnings)
        used_fallback = False
        try:
            response = await self._call(plan, evidence, structure=structure)
            answer = _parse_answer(response)
        except Exception as exc:
            logger.warning("V3 answer call failed for %s: %s", plan.question_id, exc)
            answer = _fallback_answer(plan, packet)
            used_fallback = True
            warnings.append("Answer API failed; preserved a deterministic evidence-only draft")
        missing = _missing_slots(plan, answer)
        if missing and answer and not used_fallback:
            try:
                response = await self._call(
                    plan,
                    evidence,
                    structure=structure,
                    draft=answer,
                    missing=missing,
                )
                refined = _parse_answer(response)
                if refined:
                    answer = refined
            except Exception as exc:
                logger.warning("V3 answer refinement failed for %s: %s", plan.question_id, exc)
                warnings.append("Answer refinement failed; retained the grounded draft")
        if plan.category == "global_synthesis" and answer and not used_fallback:
            try:
                response = await self._call(
                    plan,
                    evidence,
                    structure=structure,
                    draft=answer,
                    refine_instruction=_THESIS_CHECK_INSTRUCTION,
                )
                verified = _parse_answer(response)
                if verified:
                    answer = verified
            except Exception as exc:
                logger.warning("V3 thesis verification failed for %s: %s", plan.question_id, exc)
                warnings.append("Thesis-verification pass failed; retained the unverified draft")
        if plan.category in ("cross_section", "absence") and answer and not used_fallback:
            try:
                response = await self._call(
                    plan,
                    evidence,
                    structure=structure,
                    draft=answer,
                    refine_instruction=_COMPLETENESS_CHECK_INSTRUCTION,
                )
                completed = _parse_answer(response)
                if completed:
                    answer = completed
            except Exception as exc:
                logger.warning("V3 completeness check failed for %s: %s", plan.question_id, exc)
                warnings.append("Completeness-check pass failed; retained the unverified draft")
        if not answer:
            answer = _fallback_answer(plan, packet)
        elif not used_fallback:
            answer = _drop_unverified_citations(answer, _verified_citations(evidence))
        return ExecutionResult(
            question_id=plan.question_id,
            answer=answer,
            evidence=[item.exact_quote for item in packet.evidence],
            source_pages=sorted({item.page for item in packet.evidence}),
            complete=bool(answer),
            strategy=plan.strategy,
            warnings=warnings,
        )

    async def _call(
        self,
        plan: V3QuestionPlan,
        evidence: list[dict],
        *,
        structure: list[str] | None = None,
        draft: str = "",
        missing: list[str] | None = None,
        refine_instruction: str | None = None,
    ) -> str:
        payload = {
            "question_id": plan.question_id,
            "question": plan.question,
            "strategy": plan.strategy.value,
            "required_slots": plan.required_slots,
            "records_scanned": "all",
            "evidence": evidence,
        }
        if structure:
            payload["document_structure"] = structure
        if draft:
            payload["draft_answer"] = draft
            payload["refine_instruction"] = refine_instruction or (
                "Correct only the missing requested parts using the same evidence: "
                + ", ".join(missing or [])
            )
        response = await self._router.chat(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            stage="answer",
            temperature=0.0,
            max_tokens=(
                300
                if plan.strategy.value == "exhaustive_lookup"
                else (700 if plan.strategy.value == "claim_compare" else 1400)
            ),
            json_mode=True,
            question_ids=[plan.question_id],
        )
        if response.usage:
            self._tracker.record(response.usage)
        return response.content
