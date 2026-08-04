"""Evidence-grounded final synthesis with one bounded slot-refinement pass."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.v3.models import EvidencePacket, ExecutionResult, V3QuestionPlan

logger = get_logger("v3.answerer")
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_answer_system.txt"


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
    return str(data.get("answer", data.get("final_answer", ""))).strip()


def _evidence_score(item) -> tuple[int, float]:
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
    return sum(signal in text for signal in signals), item.confidence


def _bounded_evidence(plan: V3QuestionPlan, packet: EvidencePacket):
    items = list(packet.evidence)
    if plan.strategy.value != "hierarchical_synthesis":
        return items
    limit = 80 if plan.category == "cross_section" else 36
    if len(items) <= limit:
        return items
    if plan.category == "cross_section":
        return sorted(items, key=_evidence_score, reverse=True)[:limit]

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
        for item in sorted(bucket, key=_evidence_score, reverse=True)[:per_bucket]
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
    ) -> ExecutionResult:
        if not packet.evidence:
            return ExecutionResult(
                question_id=plan.question_id,
                strategy=plan.strategy,
                complete=False,
                warnings=[*packet.warnings, "No complete grounded evidence packet"],
            )
        evidence = _evidence_payload(plan, packet)
        warnings = list(packet.warnings)
        used_fallback = False
        try:
            response = await self._call(plan, evidence)
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
                    draft=answer,
                    missing=missing,
                )
                refined = _parse_answer(response)
                if refined:
                    answer = refined
            except Exception as exc:
                logger.warning("V3 answer refinement failed for %s: %s", plan.question_id, exc)
                warnings.append("Answer refinement failed; retained the grounded draft")
        if not answer:
            answer = _fallback_answer(plan, packet)
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
        draft: str = "",
        missing: list[str] | None = None,
    ) -> str:
        payload = {
            "question_id": plan.question_id,
            "question": plan.question,
            "strategy": plan.strategy.value,
            "required_slots": plan.required_slots,
            "records_scanned": "all",
            "evidence": evidence,
        }
        if draft:
            payload["draft_answer"] = draft
            payload["refine_instruction"] = (
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
