"""Batched exhaustive evidence mapping over every compiled document record."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from app.config import Settings
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.v3.compiler import all_mapping_records
from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    EvidenceCandidate,
    Strategy,
    TopicAssessment,
    V3MapResult,
    V3QuestionPlan,
)

logger = get_logger("v3.exhaustive_mapper")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_mapper_system.txt"
MAPPER_CONTRACT_VERSION = "v2"
TERMINAL_STATUSES = {"evidence_found", "no_evidence"}
STATUS_PRIORITY = {
    "llm_failed": 0,
    "parse_failed": 1,
    "uncertain": 2,
    "no_evidence": 3,
    "evidence_found": 4,
}
T = TypeVar("T")


def _chunks(items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _pack_records(
    records: list[CompiledRecord],
    *,
    max_items: int,
    max_characters: int,
) -> list[list[CompiledRecord]]:
    """Pack every record exactly once while bounding each request's source text."""
    batches: list[list[CompiledRecord]] = []
    current: list[CompiledRecord] = []
    current_characters = 0
    for record in records:
        size = len(record.text)
        if current and (
            len(current) >= max_items
            or current_characters + size > max_characters
        ):
            batches.append(current)
            current = []
            current_characters = 0
        current.append(record)
        current_characters += size
    if current:
        batches.append(current)
    return batches


def _normalize_source(text: str) -> str:
    translations = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
        }
    )
    normalized = unicodedata.normalize("NFKC", html.unescape(text)).translate(translations)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _canonical_quote(record: CompiledRecord, quote: str) -> tuple[str, int] | None:
    wanted = _normalize_source(quote)
    if len(wanted) < 8:
        return None
    literal_fragments = [
        _normalize_source(fragment)
        for fragment in re.split(r"(?:\.{3}|\u2026)", quote)
        if len(_normalize_source(fragment)) >= 30
    ]
    page_pattern = re.compile(r"\[Page (\d+)\]\s*")
    matches = list(page_pattern.finditer(record.text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(record.text)
        page_text = record.text[match.end() : end]
        normalized_page = _normalize_source(page_text)
        direct_match = wanted in normalized_page
        fragment_match = bool(literal_fragments) and all(
            fragment in normalized_page for fragment in literal_fragments
        )
        if not direct_match and not fragment_match:
            continue
        spans = [
            span.strip() for span in re.split(r"(?<=[.!?])\s+|\n\s*\n", page_text) if span.strip()
        ]
        for span in spans:
            normalized_span = _normalize_source(span)
            if wanted in normalized_span or (
                literal_fragments
                and all(fragment in normalized_span for fragment in literal_fragments)
            ):
                return span, int(match.group(1))
        if direct_match:
            return quote, int(match.group(1))
    if wanted in _normalize_source(record.text):
        return quote, record.page_start
    return None


def _topic_key(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_source(topic))


def _json_payload(raw: str) -> Any:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _bounded_confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _validated_topic_level(topic: str, level: str, quote: str) -> str:
    """Reject semantic expansions that can invert an absence conclusion."""
    if level not in {"substantive", "mention_only"}:
        return level
    if "poaching" in _topic_key(topic):
        normalized_quote = _normalize_source(quote)
        poaching_markers = (
            "poach",
            "illegal hunt",
            "illegally hunt",
            "illegal killing",
            "wildlife crime",
        )
        if not any(marker in normalized_quote for marker in poaching_markers):
            return "none"
    return level


def _exact_topic_quote(
    record: CompiledRecord,
    topic: str,
) -> tuple[str, int] | None:
    """Return a source sentence when the candidate phrase is literally present."""
    variants = [topic.strip()]
    without_article = re.sub(r"^(?:the|a|an)\s+", "", topic.strip(), flags=re.I)
    variants.append(without_article)
    variants.append(
        re.sub(r"\s+(?:designations?|status)$", "", without_article, flags=re.I)
    )
    for variant in dict.fromkeys(value for value in variants if len(value) >= 4):
        words = re.findall(r"[A-Za-z0-9]+", variant)
        if not words:
            continue
        pattern = re.compile(
            r"\b" + r"[\s\-\u2013\u2014]+".join(map(re.escape, words)) + r"\b",
            re.I,
        )
        for match in pattern.finditer(record.text):
            start_candidates = [
                record.text.rfind(delimiter, 0, match.start())
                for delimiter in (". ", "? ", "! ", "\n")
            ]
            start = max(start_candidates) + 1
            end_candidates = [
                position
                for delimiter in (". ", "? ", "! ", "\n")
                if (position := record.text.find(delimiter, match.end())) >= 0
            ]
            end = min(end_candidates) + 1 if end_candidates else len(record.text)
            quote = record.text[start:end].strip()
            folded_quote = _normalize_source(quote)
            absence_markers = (
                "does not discuss",
                "does not substantively discuss",
                "does not mention",
                "not discussed",
                "never mentioned",
                "never raised",
                "no discussion of",
                "absent from",
                "without discussing",
            )
            if any(marker in folded_quote for marker in absence_markers):
                continue
            canonical = _canonical_quote(record, quote)
            if canonical is not None:
                return canonical
    return None


class ExhaustiveMapper:
    """Map unresolved questions across all records with complete pair coverage."""

    def __init__(
        self,
        router: LLMRouter,
        usage_tracker: UsageTracker,
        settings: Settings,
    ) -> None:
        self._router = router
        self._tracker = usage_tracker
        self._settings = settings
        self._system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
        self._prompt_hash = hashlib.sha256(self._system_prompt.encode("utf-8")).hexdigest()
        self._cache_dir = settings.cache_dir / "v3"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def map_document(
        self,
        document: CompiledDocument,
        plans: list[V3QuestionPlan],
        *,
        progress_callback: Callable[[str, int, int, int], None] | None = None,
        model_override: str | None = None,
    ) -> list[V3MapResult]:
        """Return one terminal map result for every unresolved question/record pair."""
        if not plans:
            return []
        records = all_mapping_records(document)
        plan_batches = self._plan_batches(plans)
        jobs: list[tuple[list[CompiledRecord], list[V3QuestionPlan]]] = []
        for plan_batch in plan_batches:
            # Absence rows contain one assessment per candidate topic. Keeping one
            # record per call prevents the JSON matrix from reaching the output cap.
            record_batch_size = (
                1
                if all(plan.strategy == Strategy.ABSENCE_MATRIX for plan in plan_batch)
                else self._settings.record_batch_size
            )
            jobs.extend(
                (record_batch, plan_batch)
                for record_batch in _pack_records(
                    records,
                    max_items=record_batch_size,
                    max_characters=self._settings.record_batch_max_characters,
                )
            )
        total = len(jobs)
        if progress_callback:
            progress_callback("mapping", 0, total, 0)

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)

        async def run_job(
            record_batch: list[CompiledRecord],
            plan_batch: list[V3QuestionPlan],
        ) -> list[V3MapResult]:
            async with semaphore:
                return await self._map_batch(
                    document,
                    record_batch,
                    plan_batch,
                    model_override=model_override,
                )

        tasks = [asyncio.create_task(run_job(*job)) for job in jobs]
        results: list[V3MapResult] = []
        failed = 0
        for processed, task in enumerate(asyncio.as_completed(tasks), start=1):
            batch_results = await task
            results.extend(batch_results)
            failed += sum(result.status not in TERMINAL_STATUSES for result in batch_results)
            if progress_callback:
                progress_callback("mapping", processed, total, failed)
        return sorted(results, key=lambda result: (result.question_id, result.record_id))

    async def map_selected_records(
        self,
        document: CompiledDocument,
        records: list[CompiledRecord],
        plans: list[V3QuestionPlan],
        *,
        model_override: str | None = None,
    ) -> list[V3MapResult]:
        """Re-map a small lexical repair set with an optionally stronger model."""
        if not records or not plans:
            return []
        batch_size = 1 if any(
            plan.strategy == Strategy.ABSENCE_MATRIX for plan in plans
        ) else self._settings.record_batch_size
        batches = _pack_records(
            records,
            max_items=batch_size,
            max_characters=self._settings.record_batch_max_characters,
        )
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)

        async def run(batch: list[CompiledRecord]) -> list[V3MapResult]:
            async with semaphore:
                return await self._map_batch(
                    document,
                    batch,
                    plans,
                    model_override=model_override,
                )

        grouped = await asyncio.gather(*(run(batch) for batch in batches))
        return sorted(
            [result for batch in grouped for result in batch],
            key=lambda result: (result.question_id, result.record_id),
        )

    async def repair_failures(
        self,
        document: CompiledDocument,
        plans: list[V3QuestionPlan],
        results: list[V3MapResult],
        *,
        progress_callback: Callable[[str, int, int, int], None] | None = None,
        model_override: str | None = None,
    ) -> list[V3MapResult]:
        """Retry only nonterminal question/record pairs using isolated calls."""
        plan_by_id = {plan.question_id: plan for plan in plans}
        record_by_id = {record.record_id: record for record in all_mapping_records(document)}
        pending = [
            result
            for result in results
            if result.status not in TERMINAL_STATUSES
            and result.question_id in plan_by_id
            and result.record_id in record_by_id
        ]
        if not pending:
            return results

        total = len(pending)
        if progress_callback:
            progress_callback("repairing", 0, total, total)
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)

        pending_by_record: dict[str, list[V3QuestionPlan]] = {}
        for result in pending:
            plans_for_record = pending_by_record.setdefault(result.record_id, [])
            plan = plan_by_id[result.question_id]
            if plan not in plans_for_record:
                plans_for_record.append(plan)

        jobs: list[tuple[CompiledRecord, list[V3QuestionPlan]]] = []
        for record_id, record_plans in pending_by_record.items():
            absence = [plan for plan in record_plans if plan.strategy == Strategy.ABSENCE_MATRIX]
            remaining = [plan for plan in record_plans if plan.strategy != Strategy.ABSENCE_MATRIX]
            for plan_group in (absence, remaining):
                jobs.extend(
                    (record_by_id[record_id], batch)
                    for batch in _chunks(
                        plan_group,
                        self._settings.question_batch_size,
                    )
                )

        async def repair_group(
            record: CompiledRecord,
            repair_plans: list[V3QuestionPlan],
        ) -> list[V3MapResult]:
            async with semaphore:
                return await self._map_batch(
                    document,
                    [record],
                    repair_plans,
                    model_override=model_override,
                )

        tasks = [
            asyncio.create_task(repair_group(record, repair_plans)) for record, repair_plans in jobs
        ]
        repaired_by_pair: dict[tuple[str, str], V3MapResult] = {}
        failed = 0
        processed = 0
        for task in asyncio.as_completed(tasks):
            repaired_group = await task
            for repaired in repaired_group:
                repaired_by_pair[(repaired.question_id, repaired.record_id)] = repaired
                if repaired.status not in TERMINAL_STATUSES:
                    failed += 1
            processed += len(repaired_group)
            if progress_callback:
                progress_callback("repairing", processed, total, failed)

        merged: list[V3MapResult] = []
        for original in results:
            repaired = repaired_by_pair.get((original.question_id, original.record_id))
            if repaired is None:
                merged.append(original)
                continue
            if STATUS_PRIORITY.get(repaired.status, -1) >= STATUS_PRIORITY.get(original.status, -1):
                merged.append(repaired)
            else:
                merged.append(original)
        return sorted(merged, key=lambda result: (result.question_id, result.record_id))

    def _plan_batches(self, plans: list[V3QuestionPlan]) -> list[list[V3QuestionPlan]]:
        isolated = [
            plan
            for plan in plans
            if plan.strategy
            in {
                Strategy.ABSENCE_MATRIX,
                Strategy.CLAIM_COMPARE,
                Strategy.EXHAUSTIVE_LOOKUP,
            }
        ]
        remaining = [plan for plan in plans if plan not in isolated]
        batches: list[list[V3QuestionPlan]] = []
        batches.extend([[plan] for plan in isolated])
        batches.extend(_chunks(remaining, self._settings.question_batch_size))
        return batches

    async def _map_batch(
        self,
        document: CompiledDocument,
        records: list[CompiledRecord],
        plans: list[V3QuestionPlan],
        *,
        model_override: str | None,
    ) -> list[V3MapResult]:
        model = model_override or self._router.get_model("mapper")
        cache_key = self._cache_key(
            document,
            records,
            plans,
            model,
            self._prompt_hash,
        )
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        expected_pairs = {
            (plan.question_id, record.record_id) for plan in plans for record in records
        }
        absence_batch = all(plan.strategy == Strategy.ABSENCE_MATRIX for plan in plans)
        topic_rows = len(records) * sum(len(plan.candidate_topics) for plan in plans)
        max_tokens = (
            min(6000, max(2800, topic_rows * 180))
            if absence_batch
            else min(6000, max(3200, len(expected_pairs) * 220))
        )
        latest: list[V3MapResult] = []
        last_error = ""
        for attempt in range(2):
            try:
                response = await self._router.chat(
                    self._messages(records, plans, attempt=attempt),
                    stage="mapper",
                    temperature=0.0,
                    max_tokens=max_tokens,
                    json_mode=True,
                    chunk_id="+".join(record.record_id for record in records),
                    question_ids=[plan.question_id for plan in plans],
                    model_override=model_override,
                )
                if response.usage:
                    self._tracker.record(response.usage)
                latest = self._parse_response(response.content, records, plans)
            except Exception as exc:
                logger.warning("V3 mapper attempt %d failed: %s", attempt + 1, exc)
                last_error = str(exc) or type(exc).__name__
                latest = self._failed_results(
                    records,
                    plans,
                    "llm_failed",
                    last_error,
                )
                continue
            if self._is_complete(latest, expected_pairs, plans):
                self._save_cache(cache_key, latest)
                return latest
        return latest or self._failed_results(
            records,
            plans,
            "llm_failed" if last_error else "parse_failed",
            last_error or "No mapper response",
        )

    def _messages(
        self,
        records: list[CompiledRecord],
        plans: list[V3QuestionPlan],
        *,
        attempt: int,
    ) -> list[dict[str, str]]:
        record_text = "\n\n".join(
            (
                f"===== RECORD {record.record_id} =====\n"
                f"TITLE: {record.title}\n"
                f"PAGES: {record.page_start}-{record.page_end}\n"
                f"TEXT:\n{record.text}\n"
                f"===== END RECORD {record.record_id} ====="
            )
            for record in records
        )
        question_payload = [
            {
                "question_id": plan.question_id,
                "question": plan.question,
                "strategy": plan.strategy.value,
                "candidate_topics": plan.candidate_topics,
                "required_slots": plan.required_slots,
                "operations": [
                    step.model_dump(mode="json") for step in plan.operations
                ],
                "target_fields": plan.target_fields,
                "entity_hints": plan.entity_hints,
            }
            for plan in plans
        ]
        retry_note = (
            "\nThis is a schema repair attempt. Check that every pair and every "
            "candidate topic is present.\n"
            if attempt
            else ""
        )
        user = (
            f"QUESTIONS:\n{json.dumps(question_payload, ensure_ascii=False)}\n\n"
            f"RECORDS:\n{record_text}\n"
            f"{retry_note}\nReturn the exhaustive JSON result now."
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def _parse_response(
        self,
        raw: str,
        records: list[CompiledRecord],
        plans: list[V3QuestionPlan],
    ) -> list[V3MapResult]:
        record_map = {record.record_id: record for record in records}
        plan_map = {plan.question_id: plan for plan in plans}
        try:
            data = _json_payload(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return self._failed_results(records, plans, "parse_failed", "Invalid JSON")
        rows = data if isinstance(data, list) else data.get("results", [])
        if not isinstance(rows, list):
            rows = []

        parsed: dict[tuple[str, str], V3MapResult] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("question_id", ""))
            record_id = str(row.get("record_id", ""))
            if question_id not in plan_map or record_id not in record_map:
                continue
            key = (question_id, record_id)
            if key in parsed:
                continue
            parsed[key] = self._parse_row(
                row,
                record_map[record_id],
                plan_map[question_id],
            )

        results: list[V3MapResult] = []
        for plan in plans:
            for record in records:
                results.append(
                    parsed.get(
                        (plan.question_id, record.record_id),
                        V3MapResult(
                            question_id=plan.question_id,
                            record_id=record.record_id,
                            status="parse_failed",
                            error="Missing question/record result",
                        ),
                    )
                )
        return results

    def _parse_row(
        self,
        row: dict[str, Any],
        record: CompiledRecord,
        plan: V3QuestionPlan,
    ) -> V3MapResult:
        status = str(row.get("status", row.get("extraction_status", "uncertain"))).lower()
        if status not in {*TERMINAL_STATUSES, "uncertain"}:
            status = "uncertain"

        evidence: list[EvidenceCandidate] = []
        raw_evidence = row.get("evidence", row.get("evidence_items", []))
        if isinstance(raw_evidence, list):
            evidence_limit = min(
                8,
                max(2, len(plan.required_slots) + len(plan.operations) + 1),
            )
            for item in raw_evidence[:evidence_limit]:
                if not isinstance(item, dict):
                    continue
                proposed_quote = str(item.get("exact_quote", "")).strip()
                canonical = _canonical_quote(record, proposed_quote)
                if canonical is None:
                    continue
                quote, page = canonical
                role = str(item.get("role", "support"))
                entity = str(item.get("entity", item.get("entity_name", "")))
                claim = str(item.get("claim", "")).strip()
                if (
                    plan.strategy == Strategy.CLAIM_COMPARE
                    and role == "claim"
                    and plan.entity_hints
                    and not any(
                        hint.casefold() in f"{entity} {claim} {quote}".casefold()
                        or hint.split()[0].casefold() in f"{entity} {claim} {quote}".casefold()
                        for hint in plan.entity_hints
                    )
                ):
                    continue
                evidence.append(
                    EvidenceCandidate(
                        question_id=plan.question_id,
                        record_id=record.record_id,
                        claim=claim,
                        exact_quote=quote,
                        page=page,
                        role=role,
                        entity=entity,
                        field=str(item.get("field", item.get("field_name", ""))),
                        value=item.get("value", item.get("normalized_value")),
                        unit=str(item.get("unit", "")),
                        confidence=_bounded_confidence(item.get("confidence", 0.5)),
                        record_ordinal=record.ordinal,
                    )
                )

        topics: list[TopicAssessment] = []
        raw_topics = row.get("topic_assessments", row.get("topics", []))
        raw_by_topic = (
            {
                _topic_key(str(item.get("topic", ""))): item
                for item in raw_topics
                if isinstance(item, dict)
            }
            if isinstance(raw_topics, list)
            else {}
        )
        for topic in plan.candidate_topics:
            item = raw_by_topic.get(_topic_key(topic))
            exact_presence = _exact_topic_quote(record, topic)
            if item is None:
                if exact_presence is not None:
                    quote, page = exact_presence
                    topics.append(TopicAssessment(
                        question_id=plan.question_id,
                        record_id=record.record_id,
                        topic=topic,
                        level="mention_only",
                        exact_quote=quote,
                        page=page,
                    ))
                    continue
                topics.append(
                    TopicAssessment(
                        question_id=plan.question_id,
                        record_id=record.record_id,
                        topic=topic,
                        level="uncertain",
                    )
                )
                continue
            level = str(item.get("level", "uncertain")).lower()
            if level not in {"substantive", "mention_only", "none", "uncertain"}:
                level = "uncertain"
            quote = str(item.get("exact_quote", "")).strip()
            if exact_presence is not None and level in {"none", "uncertain"}:
                quote, page = exact_presence
                level = "mention_only"
            level = _validated_topic_level(topic, level, quote)
            if level == "none":
                quote = ""
            canonical = _canonical_quote(record, quote) if quote else None
            if level in {"substantive", "mention_only"} and canonical is None:
                level, quote, page = "uncertain", "", 0
            elif canonical is not None:
                quote, page = canonical
            else:
                page = 0
            topics.append(
                TopicAssessment(
                    question_id=plan.question_id,
                    record_id=record.record_id,
                    topic=topic,
                    level=level,
                    exact_quote=quote if level != "none" else "",
                    page=page or 0,
                )
            )

        if plan.strategy == Strategy.ABSENCE_MATRIX:
            if any(topic.level == "uncertain" for topic in topics):
                status = "uncertain"
            elif any(topic.level in {"substantive", "mention_only"} for topic in topics):
                status = "evidence_found"
            else:
                status = "no_evidence"
        elif evidence:
            status = "evidence_found"
        elif status == "evidence_found":
            status = "uncertain"

        return V3MapResult(
            question_id=plan.question_id,
            record_id=record.record_id,
            status=status,
            evidence=evidence,
            topics=topics,
            error=str(row.get("error", "")),
        )

    @staticmethod
    def _failed_results(
        records: list[CompiledRecord],
        plans: list[V3QuestionPlan],
        status: str,
        error: str,
    ) -> list[V3MapResult]:
        return [
            V3MapResult(
                question_id=plan.question_id,
                record_id=record.record_id,
                status=status,
                error=error,
            )
            for plan in plans
            for record in records
        ]

    @staticmethod
    def _is_complete(
        results: list[V3MapResult],
        expected_pairs: set[tuple[str, str]],
        plans: list[V3QuestionPlan],
    ) -> bool:
        if {(result.question_id, result.record_id) for result in results} != expected_pairs:
            return False
        plan_map = {plan.question_id: plan for plan in plans}
        for result in results:
            if result.status not in TERMINAL_STATUSES:
                return False
            plan = plan_map[result.question_id]
            if plan.strategy == Strategy.ABSENCE_MATRIX:
                if len(result.topics) != len(plan.candidate_topics):
                    return False
                if any(topic.level == "uncertain" for topic in result.topics):
                    return False
        return True

    @staticmethod
    def _cache_key(
        document: CompiledDocument,
        records: list[CompiledRecord],
        plans: list[V3QuestionPlan],
        model: str,
        prompt_hash: str,
    ) -> str:
        payload = {
            "compiler": document.compiler_version,
            "mapper_contract": MAPPER_CONTRACT_VERSION,
            "model": model,
            "records": [
                [record.record_id, hashlib.sha256(record.text.encode()).hexdigest()]
                for record in records
            ],
            "plans": [plan.model_dump(mode="json") for plan in plans],
            "prompt_hash": prompt_hash,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode()).hexdigest()[:28]

    def _load_cache(self, key: str) -> list[V3MapResult] | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return [
                V3MapResult.model_validate(item)
                for item in json.loads(path.read_text(encoding="utf-8"))
            ]
        except (OSError, ValueError, TypeError):
            return None

    def _save_cache(self, key: str, results: list[V3MapResult]) -> None:
        path = self._cache_dir / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(
                    [result.model_dump(mode="json") for result in results],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not write V3 mapper cache: %s", exc)
