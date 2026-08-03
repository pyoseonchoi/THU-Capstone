"""Claim-level verification of generated answers.

Breaks the draft answer into claims and verifies each against
the evidence ledger and operation results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.evidence_context import format_evidence_context
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.schemas import (
    ClaimStatus,
    ClaimVerification,
    EvidenceItem,
    OperationResult,
)

logger = get_logger("verification.claim_verifier")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "verifier_system.txt"
MAX_REPAIR_ATTEMPTS = 2


class ClaimVerifier:
    """Verifies claims in a generated answer against evidence."""

    def __init__(self, router: LLMRouter, usage_tracker: UsageTracker):
        self._router = router
        self._tracker = usage_tracker
        self._system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    async def verify(
        self,
        draft_answer: str,
        evidence_items: list[EvidenceItem],
        operation_result: Optional[OperationResult],
        question_id: str,
    ) -> tuple[list[ClaimVerification], bool]:
        """Verify claims in the draft answer.

        Returns:
            Tuple of (claim verifications, needs_repair).
        """
        # Build evidence summary for context
        preferred_ids = (
            operation_result.supporting_evidence_ids
            if operation_result
            else []
        )
        evidence_summary = format_evidence_context(
            evidence_items,
            lambda e: (
                f"- [{e.evidence_id}] {e.entity_name}: {e.claim} "
                f"(pages {e.page_start}-{e.page_end}, confidence {e.confidence})"
            ),
            preferred_ids=preferred_ids,
        )

        op_summary = ""
        if operation_result:
            op_summary = json.dumps(
                operation_result.model_dump(mode="json"),
                ensure_ascii=False,
            )

        user_msg = (
            f"DRAFT ANSWER:\n{draft_answer}\n\n"
            f"EVIDENCE:\n{evidence_summary}\n\n"
            f"OPERATION RESULT:\n{op_summary}\n\n"
            f"Verify each claim in the draft answer."
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = await self._router.chat(
            messages,
            stage="verifier",
            json_mode=True,
            max_tokens=2048,
            question_ids=[question_id],
        )
        if response.usage:
            self._tracker.record(response.usage)

        claims, needs_repair = self._parse_verification(response.content)
        return claims, needs_repair

    def _parse_verification(
        self, raw: str
    ) -> tuple[list[ClaimVerification], bool]:
        """Parse verifier output into ClaimVerification objects."""
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
        except Exception:
            logger.warning("Verifier returned invalid JSON")
            return [], False

        claims: list[ClaimVerification] = []
        for c in data.get("claims", []):
            try:
                status = ClaimStatus(c.get("status", "UNSUPPORTED"))
            except ValueError:
                status = ClaimStatus.UNSUPPORTED

            claims.append(ClaimVerification(
                claim=c.get("claim", ""),
                status=status,
                supporting_evidence_ids=c.get("supporting_evidence_ids", []),
                operation_id=c.get("operation_id", ""),
                correction=c.get("correction", ""),
            ))

        needs_repair = data.get("overall_status", "PASS") == "NEEDS_REPAIR"
        return claims, needs_repair

    async def verify_and_repair(
        self,
        draft_answer: str,
        evidence_items: list[EvidenceItem],
        operation_result: Optional[OperationResult],
        question_id: str,
    ) -> tuple[str, list[ClaimVerification]]:
        """Verify and optionally repair the answer.

        Returns:
            Tuple of (final_answer, verifications).
        """
        claims, needs_repair = await self.verify(
            draft_answer, evidence_items, operation_result, question_id
        )

        if not needs_repair:
            return draft_answer, claims

        # Attempt repair (limited attempts to avoid loops)
        for attempt in range(MAX_REPAIR_ATTEMPTS):
            unsupported = [
                c for c in claims if c.status == ClaimStatus.UNSUPPORTED
            ]
            if not unsupported:
                break

            logger.info(
                "Repair attempt %d: %d unsupported claims",
                attempt + 1, len(unsupported),
            )

            # For now, just strip unsupported claims from the answer
            repaired = draft_answer
            for c in unsupported:
                if c.claim in repaired:
                    repaired = repaired.replace(c.claim, "").strip()

            # Re-verify
            claims, needs_repair = await self.verify(
                repaired, evidence_items, operation_result, question_id
            )
            draft_answer = repaired

            if not needs_repair:
                break

        return draft_answer, claims
