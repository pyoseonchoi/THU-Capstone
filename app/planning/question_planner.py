"""Question planner: transforms natural-language questions into QueryPlans.

Uses the configured planner LLM to generate structured plans,
validates with Pydantic, and retries once on validation failure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from app.exceptions import StructuredOutputError
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.schemas import Operator, QueryPlan, QuestionRequest

logger = get_logger("planning.question_planner")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "planner_system.txt"


class QuestionPlanner:
    """Transforms questions into structured QueryPlans via LLM."""

    def __init__(self, router: LLMRouter, usage_tracker: UsageTracker):
        self._router = router
        self._tracker = usage_tracker
        self._system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    async def plan(self, question: QuestionRequest) -> QueryPlan:
        """Generate a QueryPlan for a single question.

        Retries once with validation error feedback if parsing fails.
        """
        user_msg = (
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n\n"
            "Produce the JSON query plan."
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = await self._router.chat(
            messages,
            stage="planner",
            json_mode=True,
            max_tokens=2048,
            question_ids=[question.question_id],
        )
        if response.usage:
            self._tracker.record(response.usage)

        plan = self._parse_plan(response.content, question)
        if plan is not None:
            return plan

        # Retry with validation error feedback
        logger.warning("First plan parse failed, retrying with feedback")
        retry_msg = (
            f"Your previous output was not valid. Error: could not parse into QueryPlan.\n"
            f"Raw output: {response.content[:500]}\n\n"
            f"Please produce valid JSON matching the schema exactly."
        )
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": retry_msg})

        response2 = await self._router.chat(
            messages,
            stage="planner",
            json_mode=True,
            max_tokens=2048,
            question_ids=[question.question_id],
        )
        if response2.usage:
            self._tracker.record(response2.usage)

        plan = self._parse_plan(response2.content, question)
        if plan is not None:
            return plan

        raise StructuredOutputError(
            "Failed to parse planner output after retry",
            raw_output=response2.content,
        )

    def _parse_plan(
        self, raw: str, question: QuestionRequest
    ) -> Optional[QueryPlan]:
        """Try to parse raw LLM output into a QueryPlan."""
        try:
            text = raw.strip()
            if "```" in text:
                text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
                text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

            try:
                data = json.loads(text.strip())
            except json.JSONDecodeError:
                match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                else:
                    raise

            if not isinstance(data, dict):
                return None

            # Ensure question_id matches
            data["question_id"] = question.question_id
            data["original_question"] = question.question

            # Clean None values in nested dicts/lists
            def _clean_nulls(o):
                if o is None:
                    return ""
                if isinstance(o, dict):
                    return {k: _clean_nulls(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_clean_nulls(v) for v in o]
                return o

            data = _clean_nulls(data)

            # Normalize operator string to valid Enum
            if "operator" in data and isinstance(data["operator"], str):
                op_str = data["operator"].upper().strip()
                try:
                    data["operator"] = Operator(op_str)
                except ValueError:
                    data["operator"] = (
                        Operator.LOOKUP
                        if "LIST" in op_str or "FIND" in op_str
                        else Operator.GENERAL_SYNTHESIS
                    )

            plan = QueryPlan(**data)
            return plan
        except Exception as exc:
            logger.warning("Plan parse error: %s (raw snippet: %s)", exc, raw[:200])
            return None

    async def plan_batch(
        self, questions: list[QuestionRequest]
    ) -> list[QueryPlan]:
        """Plan multiple questions sequentially."""
        plans: list[QueryPlan] = []
        for q in questions:
            plan = await self.plan(q)
            plans.append(plan)
        return plans
