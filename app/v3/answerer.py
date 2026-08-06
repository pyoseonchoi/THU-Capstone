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


def _is_malformed_json(text: str) -> bool:
    """Check if text looks like broken/incomplete JSON."""
    text = text.strip()
    if text.startswith(('{', '[')):
        open_count = text.count('{') + text.count('[')
        close_count = text.count('}') + text.count(']')
        if open_count != close_count:
            return True
        try:
            json.loads(text)
            return False
        except json.JSONDecodeError:
            return True
    return False


def _parse_answer(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # Check for malformed JSON early
    if _is_malformed_json(text):
        return ""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return text
        if _is_malformed_json(match.group(0)):
            return ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return text
    return str(data.get("answer", data.get("final_answer", ""))).strip()


def _evidence_payload(packet: EvidencePacket) -> list[dict]:
    """Represent every reduced item within a bounded per-item envelope."""
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
            "claim": item.claim[:600],
            "exact_quote": item.exact_quote[:900],
        }
        for index, item in enumerate(packet.evidence, start=1)
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
        evidence = _evidence_payload(packet)
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

    async def finalize_deterministic(
        self,
        plan: V3QuestionPlan,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Use the answer model only to polish a Python-computed result."""
        evidence = [
            {
                "evidence_id": f"D{index:03d}",
                "exact_quote": item[:900],
            }
            for index, item in enumerate(result.evidence[:10], start=1)
            if item
        ]
        payload = {
            "question_id": plan.question_id,
            "question": plan.question,
            "strategy": plan.strategy.value,
            "python_computed_answer": result.answer,
            "source_pages": result.source_pages,
            "evidence": evidence,
            "instruction": (
                "Rewrite the Python-computed answer as a concise final answer. "
                "Do not perform arithmetic, recount items, change numeric values, "
                "change named entities, add unsupported facts, or omit required "
                "parts. Return JSON with an 'answer' field only."
            ),
        }
        try:
            response = await self._router.chat(
                [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                stage="answer",
                temperature=0.0,
                max_tokens=1200,
                json_mode=True,
                question_ids=[plan.question_id],
            )
            if response.usage:
                self._tracker.record(response.usage)
            answer = _parse_answer(response.content) or result.answer
            if not answer:
                return result.model_copy(
                    update={
                        "warnings": [
                            *result.warnings,
                            "LLM finalization rejected (malformed); retained Python answer.",
                        ]
                    }
                )
            return result.model_copy(
                update={
                    "answer": answer,
                    "warnings": [
                        *result.warnings,
                        "Deterministic Python answer finalized by LLM.",
                    ],
                }
            )
        except Exception as exc:
            logger.warning(
                "Deterministic finalization failed for %s: %s",
                plan.question_id,
                exc,
            )
            return result.model_copy(
                update={
                    "warnings": [
                        *result.warnings,
                        "LLM finalization failed; retained deterministic Python answer.",
                    ]
                }
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
            max_tokens=2500,
            json_mode=True,
            question_ids=[plan.question_id],
        )
        if response.usage:
            self._tracker.record(response.usage)
        return response.content
