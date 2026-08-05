"""Route each question to the operation that answers it.

The deterministic executors used to recognise their own questions by the
phrases one evaluation happened to use — "largest in its country", "different
units", "first recorded". Those phrasings answer that evaluation and nothing
else: a document asking the same question in its own words reaches no executor
at all. The shape of a question is a question of meaning, so a model reads it
and names the operation, bounded to the fields and groups the document holds.

Nothing here knows what the document is about. Every argument a shape takes is
either copied from the question or checked against the compiled registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.schemas import QuestionRequest
from app.v3.field_binder import field_descriptions, table_descriptions
from app.v3.models import CompiledDocument, QuestionShape, ShapePlan
from app.v3.source_text import json_payload

logger = get_logger("v3.shape_classifier")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_shape_system.txt"
CLASSIFIER_CONTRACT_VERSION = "v1"

_COMPARATORS = {"gte", "gt", "lte", "lt"}
_DIRECTIONS = {"max", "min"}
# Shapes whose arguments Python cannot supply for itself, and which therefore
# only route when the model states them.
_REQUIRED_ARGUMENTS: dict[QuestionShape, tuple[str, ...]] = {
    QuestionShape.COUNT_BY_THRESHOLD: ("field", "comparator", "threshold"),
    QuestionShape.EXTREMUM: ("field", "direction"),
    QuestionShape.UNIT_OUTLIER: ("field",),
    QuestionShape.CLAIM_CONFLICT: ("field",),
    QuestionShape.DATED_EVENT: ("event",),
}


def document_groups(document: CompiledDocument) -> list[str]:
    """Return the values the registry files its records under."""
    return sorted({record.country for record in document.records if record.country})


class ShapeClassifier:
    """Name the operation that answers each question, or decline to."""

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
        self._cache_dir = settings.cache_dir / "v3" / "shapes"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def classify(
        self,
        questions: list[QuestionRequest],
        document: CompiledDocument,
    ) -> dict[str, ShapePlan]:
        """Return question id to shape plan, omitting questions it declines."""
        if not self._settings.enable_shape_classification or not questions:
            return {}
        fields = field_descriptions(document) + table_descriptions(document)
        groups = document_groups(document)
        if not fields and not groups and not document.contents_entries:
            # With no fields, no groups and no contents there is no operation
            # to route to, and every question reaches the exhaustive scan.
            return {}
        catalog = self._catalog(document, fields, groups)
        key = self._cache_key(questions, catalog)
        payload = self._load_cache(key)
        if payload is None:
            payload = await self._ask(questions, catalog)
            if payload is None:
                return {}
            self._save_cache(key, payload)

        allowed_fields = {field for field, _, _ in fields}
        allowed_groups = {group.casefold(): group for group in groups}
        asked = {question.question_id for question in questions}
        plans: dict[str, ShapePlan] = {}
        for item in payload.get("questions", []):
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id", "")).strip()
            if question_id not in asked:
                continue
            plan = _validated_plan(item, allowed_fields, allowed_groups)
            if plan is not None:
                plans[question_id] = plan
        routed = sum(1 for plan in plans.values() if plan.routes)
        logger.info(
            "Shaped %d/%d questions; %d route to an executor",
            len(plans),
            len(questions),
            routed,
        )
        return plans

    @staticmethod
    def _catalog(
        document: CompiledDocument,
        fields: list[tuple[str, str, int]],
        groups: list[str],
    ) -> str:
        noun = document.entity_label or "record"
        lines = [
            f"The document holds {len(document.records)} records, each a {noun}.",
            "",
            "FIELDS its records report:",
        ]
        lines += [
            f"- {field} | document wording: \"{label}\" | "
            f"reported by {count} of {len(document.records)} records"
            for field, label, count in fields
        ] or ["- (none)"]
        lines += ["", "GROUP values it files records under:"]
        lines.append("- " + (", ".join(groups) if groups else "(none)"))
        if document.contents_entries:
            lines += ["", "The document also has a parsed contents list."]
        return "\n".join(lines)

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
                            f"DOCUMENT\n{catalog}\n\n"
                            f"QUESTIONS\n{listed}\n\n"
                            "Return the routing JSON now."
                        ),
                    },
                ],
                stage="planner",
                temperature=0.0,
                max_tokens=2000,
                json_mode=True,
                question_ids=[question.question_id for question in questions],
            )
            if response.usage:
                self._tracker.record(response.usage)
            payload = json_payload(response.content)
        except Exception as exc:
            logger.warning("Shape classification failed: %s", exc)
            return None
        return payload if isinstance(payload, dict) else None

    def _cache_key(self, questions: list[QuestionRequest], catalog: str) -> str:
        payload = json.dumps(
            {
                "contract": CLASSIFIER_CONTRACT_VERSION,
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
            logger.warning("Could not write shape cache: %s", exc)


def _validated_plan(
    item: dict[str, Any],
    allowed_fields: set[str],
    allowed_groups: dict[str, str],
) -> ShapePlan | None:
    """Build a plan from one routing row, keeping only supportable arguments."""
    try:
        shape = QuestionShape(str(item.get("shape", "")).strip().casefold())
    except ValueError:
        return None

    field = str(item.get("field", "")).strip()
    if field and field not in allowed_fields:
        # A field the document does not report would send an executor looking
        # for a column that is not there.
        field = ""

    groups: list[str] = []
    raw_groups = item.get("groups", [])
    if isinstance(raw_groups, list):
        for value in raw_groups:
            canonical = allowed_groups.get(str(value).strip().casefold())
            if canonical and canonical not in groups:
                groups.append(canonical)

    raw_terms = item.get("search_terms", [])
    search_terms = [
        str(term).strip()[:40]
        for term in (raw_terms if isinstance(raw_terms, list) else [])
        if str(term).strip()
    ][:10]
    comparator = str(item.get("comparator", "")).strip().casefold()
    direction = str(item.get("direction", "")).strip().casefold()
    threshold = item.get("threshold")
    plan = ShapePlan(
        shape=shape,
        field=field,
        groups=groups,
        comparator=comparator if comparator in _COMPARATORS else "",
        threshold=float(threshold) if isinstance(threshold, (int, float)) else None,
        direction=direction if direction in _DIRECTIONS else "",
        claim_scope=str(item.get("claim_scope", "")).strip()[:80],
        search_terms=search_terms,
        event=str(item.get("event", "")).strip()[:120],
        subject=str(item.get("subject", "")).strip()[:120],
        confident=bool(item.get("confident", False)),
    )

    # A shape missing an argument it cannot run without is not a route.
    for name in _REQUIRED_ARGUMENTS.get(shape, ()):
        if not getattr(plan, name):
            plan.confident = False
            break
    return plan
