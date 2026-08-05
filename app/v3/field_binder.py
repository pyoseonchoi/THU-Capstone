"""Bind questions to the compiled fields that answer them.

Which field a question asks about is a question of meaning, not of wording. A
document may report "Annual visiting researchers" for what a question calls
"visitors per year", while a question asking for a figure "per year" shares the
word "year" with a field recording when something was founded. Lexical scoring
gets the second case confidently wrong, so the model does the matching and the
document's own field list bounds what it may answer.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.schemas import QuestionRequest
from app.v3.models import CompiledDocument
from app.v3.source_text import json_payload

logger = get_logger("v3.field_binder")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_bind_system.txt"
BINDER_CONTRACT_VERSION = "v1"


def field_descriptions(document: CompiledDocument) -> list[tuple[str, str, int]]:
    """Describe each field by its identifier, document wording, and coverage."""
    labels: dict[str, Counter[str]] = {}
    holders: dict[str, set[str]] = {}
    for record in document.records:
        for fact in record.number_facts:
            labels.setdefault(fact.field, Counter())[fact.label] += 1
            holders.setdefault(fact.field, set()).add(record.record_id)
    described = [
        (field, labels[field].most_common(1)[0][0], len(holders[field]))
        for field in document.field_catalog
        if field in labels
    ]
    # A field only a couple of records report cannot answer a question about
    # the document as a whole, and listing it invites a wrong match.
    minimum = max(2, round(len(document.records) * 0.2))
    return sorted(
        (item for item in described if item[2] >= minimum),
        key=lambda item: (-item[2], item[0]),
    )


def table_descriptions(document: CompiledDocument) -> list[tuple[str, str, int]]:
    """Describe each compiled table column the way its own header names it.

    A document whose figures live in tables rather than in per-record fact
    cards has nothing in the record catalog to bind a question to, so a
    question about one of its columns reaches no operation. The columns
    themselves are already named — the compiler read those names out of the
    table's own header — so they belong beside the record fields.
    """
    described: list[tuple[str, str, int]] = []
    for table in document.tables:
        if not table.trusted or not table.rows:
            continue
        for column in table.columns:
            filled = sum(
                1 for row in table.rows
                if isinstance(row.values.get(column), (int, float))
            )
            if filled >= max(2, round(len(table.rows) * 0.5)):
                described.append((column, f"{table.title} — {column}", filled))
    return sorted(described, key=lambda item: (-item[2], item[0]))


class FieldBinder:
    """Resolve each question to at most one compiled field."""

    def __init__(
        self,
        router: LLMRouter,
        tracker: UsageTracker,
        settings: Settings,
    ) -> None:
        self._router = router
        self._tracker = tracker
        self._settings = settings
        self._system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
        self._prompt_hash = hashlib.sha256(
            self._system_prompt.encode("utf-8")
        ).hexdigest()
        self._cache_dir = settings.cache_dir / "v3" / "bindings"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def bind(
        self,
        questions: list[QuestionRequest],
        document: CompiledDocument,
    ) -> dict[str, str] | None:
        """Return question id to field, or None when nothing was decided.

        An empty mapping and a missing mapping mean different things: the
        first says the model read the field list and found none that answers
        the question, which callers must respect. None says no verdict was
        reached at all, leaving callers free to fall back.
        """
        if not self._settings.enable_field_binding or not questions:
            return None
        fields = field_descriptions(document) + table_descriptions(document)
        if not fields:
            return None
        allowed = {field for field, _, _ in fields}
        catalog = "\n".join(
            f"- {field} | document wording: \"{label}\" | "
            f"reported by {count} of {len(document.records)} records"
            for field, label, count in fields
        )
        key = self._cache_key(questions, catalog)
        payload = self._load_cache(key)
        if payload is None:
            payload = await self._ask(questions, catalog)
            if payload is None:
                return None
            self._save_cache(key, payload)
        bindings: dict[str, str] = {}
        asked = {question.question_id for question in questions}
        for item in payload.get("bindings", []):
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id", "")).strip()
            field = str(item.get("field", "")).strip()
            if question_id in asked and field in allowed:
                bindings[question_id] = field
        logger.info("Bound %d/%d questions to a field", len(bindings), len(questions))
        return bindings

    async def _ask(
        self,
        questions: list[QuestionRequest],
        catalog: str,
    ) -> dict[str, Any] | None:
        listed = "\n".join(
            f"- {question.question_id}: {question.question}" for question in questions
        )
        try:
            response = await self._router.chat(
                [
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"AVAILABLE FIELDS\n{catalog}\n\n"
                            f"QUESTIONS\n{listed}\n\n"
                            "Return the bindings JSON now."
                        ),
                    },
                ],
                stage="planner",
                temperature=0.0,
                max_tokens=1200,
                json_mode=True,
                question_ids=[question.question_id for question in questions],
            )
            if response.usage:
                self._tracker.record(response.usage)
            payload = json_payload(response.content)
        except Exception as exc:
            logger.warning("Field binding failed: %s", exc)
            return None
        return payload if isinstance(payload, dict) else None

    def _cache_key(self, questions: list[QuestionRequest], catalog: str) -> str:
        payload = json.dumps(
            {
                "contract": BINDER_CONTRACT_VERSION,
                "model": self._router.get_model("planner"),
                "prompt": self._prompt_hash,
                "catalog": catalog,
                "questions": [
                    [question.question_id, question.question] for question in questions
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:28]

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _save_cache(self, key: str, payload: dict[str, Any]) -> None:
        path = self._cache_dir / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not write binding cache: %s", exc)
