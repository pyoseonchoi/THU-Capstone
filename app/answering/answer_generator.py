"""Grounded answer generator.

Generates the final answer from evidence, operation results,
and coverage information. Uses only supplied evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.evidence_context import format_evidence_context
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.schemas import (
    CoverageReport,
    EvidenceItem,
    OperationResult,
    QueryPlan,
)

logger = get_logger("answering.answer_generator")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "answer_system.txt"


class AnswerGenerator:
    """Generates grounded answers from evidence and operation results."""

    def __init__(self, router: LLMRouter, usage_tracker: UsageTracker):
        self._router = router
        self._tracker = usage_tracker
        self._system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    async def generate(
        self,
        plan: QueryPlan,
        operation_result: OperationResult,
        evidence_items: list[EvidenceItem],
        coverage: CoverageReport,
    ) -> tuple[str, str]:
        """Generate final answer and answer-with-evidence.

        Args:
            plan: The query plan.
            operation_result: Deterministic operation result.
            evidence_items: Supporting evidence.
            coverage: Coverage report.

        Returns:
            Tuple of (final_answer, answer_with_evidence).
        """
        # Build structured context for the LLM
        evidence_summary = format_evidence_context(
            evidence_items,
            lambda e: (
                f"- [{e.evidence_id}] {e.entity_name}: "
                f"{e.field_name}={e.raw_value} "
                f"(pages {e.page_start}-{e.page_end}, "
                f"quote: \"{e.exact_quote[:120]}\")"
            ),
            preferred_ids=operation_result.supporting_evidence_ids,
        )

        op_summary = json.dumps(
            operation_result.model_dump(mode="json"),
            ensure_ascii=False,
        )

        coverage_summary = (
            f"Coverage: {coverage.coverage_percentage}% "
            f"({coverage.successful_mappings + coverage.no_evidence_chunks}/"
            f"{coverage.total_chunks} chunks)\n"
            f"Entities found: {coverage.detected_entity_count}\n"
            f"Required coverage: {coverage.required_percentage}%\n"
            f"Requirement met: {coverage.requirement_met}\n"
            f"Warnings: {coverage.warnings}"
        )

        user_msg = (
            f"QUESTION: {plan.original_question}\n\n"
            f"QUERY PLAN:\n"
            f"Operator: {plan.operator.value}\n"
            f"Entity type: {plan.entity_type}\n"
            f"Target fields: {plan.target_fields}\n"
            f"Conditions: {[c.model_dump() for c in plan.conditions]}\n\n"
            f"OPERATION RESULT:\n{op_summary}\n\n"
            f"EVIDENCE:\n{evidence_summary}\n\n"
            f"COVERAGE:\n{coverage_summary}\n\n"
            f"Generate the final answer as JSON with 'final_answer' and 'answer_with_evidence'."
        )

        if operation_result.warnings:
            user_msg += "\n\nWARNINGS:\n" + "\n".join(operation_result.warnings)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = await self._router.chat(
            messages,
            stage="answer",
            json_mode=True,
            max_tokens=2048,
            question_ids=[plan.question_id],
        )
        if response.usage:
            self._tracker.record(response.usage)

        return self._parse_response(response.content, operation_result)

    def _parse_response(
        self,
        raw: str,
        operation_result: OperationResult,
    ) -> tuple[str, str]:
        """Parse the answer LLM response."""
        try:
            text = raw.strip()
            if "```" in text:
                import re as re_mod
                text = re_mod.sub(
                    r"^```(?:json)?\s*",
                    "",
                    text,
                    flags=re_mod.IGNORECASE | re_mod.MULTILINE,
                )
                text = re_mod.sub(r"\s*```$", "", text, flags=re_mod.MULTILINE)
            try:
                data = json.loads(text.strip())
            except json.JSONDecodeError:
                import re as re_mod
                match = re_mod.search(r"(\{.*\}|\[.*\])", text, re_mod.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                else:
                    raise
            final = data.get("final_answer", "")
            with_evidence = data.get("answer_with_evidence", final)
            return final, with_evidence
        except Exception:
            # Fallback: use the raw response as the answer
            logger.warning("Could not parse answer JSON, using raw text")
            return raw.strip(), raw.strip()
