"""Schema-free per-record fact extraction validated against the source.

The deterministic compiler can only bind numbers whose layout it recognises,
so a document whose fact cards differ from that layout yields a half-populated
registry — and a count or extremum over a half-populated registry is silently
wrong. This stage asks the cheap model to read each record once and report the
measurements it states, then keeps only the facts it can prove against the
record text.

Nothing here knows what the document is about. Metric names come from the
document's own wording, and the field catalog is whatever those metrics
cluster into.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.reduction.normalizer import parse_number
from app.v3.compiler import all_mapping_records, field_id
from app.v3.models import CompiledDocument, CompiledRecord, NumberFact
from app.v3.source_text import json_payload, normalize_source

logger = get_logger("v3.record_profiler")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_profile_system.txt"
PROFILER_CONTRACT_VERSION = "v1"

_SCALE_WORDS = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
_NUMBER_IN_TEXT_RE = re.compile(
    r"-?\d[\d,. ]*(?:\s*(?:thousand|million|billion|trillion))?",
    re.IGNORECASE,
)
ProgressCallback = Callable[[str, int, int, int], None]


# The compiler owns field identity. Deriving it separately here would file the
# model's wording for a measurement under a different field than the layout
# parser used for the same measurement, splitting its coverage in half.
metric_field = field_id


def _quote_numbers(quote: str) -> list[float]:
    """Return every number the quote states, applying written scale words."""
    values: list[float] = []
    for match in _NUMBER_IN_TEXT_RE.finditer(quote):
        token = match.group(0).strip()
        scale = 1
        for word, factor in _SCALE_WORDS.items():
            if token.casefold().endswith(word):
                token = token[: -len(word)].strip()
                scale = factor
                break
        parsed = parse_number(token)
        if parsed is not None:
            values.append(parsed * scale)
            if scale != 1:
                values.append(parsed)
    return values


def _value_supported(value: float, quote: str) -> bool:
    """Require the reported number to be present in its own quote."""
    for candidate in _quote_numbers(quote):
        if candidate == value:
            return True
        if value and abs(candidate - value) <= abs(value) * 1e-6:
            return True
    return False


class RecordProfiler:
    """Extract and verify one structured profile per compiled record."""

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
        self._cache_dir = settings.cache_dir / "v3" / "profiles"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def profile(
        self,
        document: CompiledDocument,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> CompiledDocument:
        """Enrich every record with verified facts, titles, and categories."""
        if not self._settings.enable_record_profiling or not _needs_profiling(document):
            return document
        records = all_mapping_records(document)
        if not records:
            return document

        # Each record is read independently, so the model cannot know how it
        # named a measurement elsewhere. A first pass over a spread-out sample
        # induces the document's own metric vocabulary, which the full pass
        # reuses so the same measurement lands in one field.
        vocabulary = await self._induce_vocabulary(records)

        total = len(records)
        if progress_callback:
            progress_callback("profiling", 0, total, 0)
        profiles, failed = await self._profile_all(
            records,
            vocabulary,
            lambda done, bad: progress_callback and
            progress_callback("profiling", done, total, bad),
        )

        by_id = {record.record_id: record for record in records}
        for record_id, payload in profiles.items():
            self._apply(by_id[record_id], payload)

        document.profiled_records = sorted(profiles)
        document.field_catalog = _discovered_catalog(
            document.records,
            self._settings.profile_min_coverage,
        )
        document.registry_signals = {
            **document.registry_signals,
            "profiled_records": len(profiles),
            "profile_failures": failed,
        }
        if failed:
            document.warnings.append(
                f"{failed} records could not be profiled; "
                "deterministic reduction stays disabled for missing fields"
            )
        logger.info(
            "Profiled %d/%d records; catalog=%s",
            len(profiles),
            total,
            document.field_catalog,
        )
        return document

    async def _profile_all(
        self,
        records: list[CompiledRecord],
        vocabulary: list[str],
        report: Callable[[int, int], Any],
    ) -> tuple[dict[str, dict[str, Any]], int]:
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)

        async def run(record: CompiledRecord) -> tuple[str, dict[str, Any] | None]:
            async with semaphore:
                return record.record_id, await self._profile_record(record, vocabulary)

        tasks = [asyncio.create_task(run(record)) for record in records]
        profiles: dict[str, dict[str, Any]] = {}
        failed = 0
        for done, task in enumerate(asyncio.as_completed(tasks), start=1):
            record_id, payload = await task
            if payload is None:
                failed += 1
            else:
                profiles[record_id] = payload
            report(done, failed)
        return profiles, failed

    async def _induce_vocabulary(self, records: list[CompiledRecord]) -> list[str]:
        """Learn the document's recurring metric names from a spread sample."""
        size = min(self._settings.profile_vocabulary_sample, len(records))
        if size < 2:
            return []
        step = max(1, len(records) // size)
        sample = records[::step][:size]
        profiles, _ = await self._profile_all(sample, [], lambda done, bad: None)
        counts: Counter[str] = Counter()
        spelling: dict[str, str] = {}
        for payload in profiles.values():
            metrics = {
                str(item.get("metric", "")).strip()
                for item in payload.get("facts", [])
                if isinstance(item, dict) and str(item.get("metric", "")).strip()
            }
            for metric in metrics:
                key = metric_field(metric)
                if not key:
                    continue
                counts[key] += 1
                spelling.setdefault(key, metric)
        vocabulary = [
            spelling[key] for key, count in counts.most_common() if count >= 2
        ]
        logger.info("Induced metric vocabulary from %d records: %s", len(sample), vocabulary)
        return vocabulary

    def _apply(self, record: CompiledRecord, payload: dict[str, Any]) -> None:
        """Merge verified profile output into a compiled record."""
        source = normalize_source(record.text)
        name = str(payload.get("name", "")).strip()
        if name and normalize_source(name) in source:
            unresolved = not record.title or record.title.startswith("Record ")
            if unresolved:
                record.title = name[:120]
        category = str(payload.get("category", "")).strip()
        if category and not record.country and normalize_source(category) in source:
            record.country = category[:80]

        verified = self._facts(record, payload)
        if not verified:
            return
        # The layout parser and the model name metrics differently, so mixing
        # both would split one measurement across two fields. Once a record is
        # profiled, the model's naming is the single authority for it, and any
        # layout-parsed metric it did not report is carried over under that
        # record's own wording.
        claimed = {fact.field for fact in verified}
        carried = [
            fact
            for fact in record.number_facts
            if fact.field not in claimed
            and not any(fact.value == item.value for item in verified)
        ]
        record.number_facts = verified + carried

    def _facts(
        self,
        record: CompiledRecord,
        payload: dict[str, Any],
    ) -> list[NumberFact]:
        raw_facts = payload.get("facts", [])
        if not isinstance(raw_facts, list):
            return []
        source = normalize_source(record.text)
        verified: list[NumberFact] = []
        seen: set[tuple[str, str]] = set()
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", "")).strip()
            quote = str(item.get("quote", "")).strip()
            if not metric or not quote:
                continue
            if normalize_source(quote) not in source:
                continue
            raw_value = item.get("value")
            value = (
                float(raw_value)
                if isinstance(raw_value, (int, float))
                else parse_number(str(raw_value or ""))
            )
            if value is None or not _value_supported(value, quote):
                continue
            field = metric_field(metric)
            if not field:
                continue
            subject = str(item.get("subject", "")).strip()
            key = (field, subject.casefold())
            if key in seen:
                continue
            seen.add(key)
            verified.append(NumberFact(
                record_id=record.record_id,
                field=field,
                label=metric,
                value=value,
                raw_value=str(item.get("value", value)),
                unit=str(item.get("unit", "")).strip(),
                subject=subject,
                page=_quote_page(record, quote),
                quote=quote,
            ))
        return verified

    async def _profile_record(
        self,
        record: CompiledRecord,
        vocabulary: list[str],
    ) -> dict[str, Any] | None:
        model = self._router.get_model("mapper")
        key = self._cache_key(record, model, vocabulary)
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        known = (
            "Metrics already seen in this document: "
            + "; ".join(vocabulary)
            + "\nReuse one of these metric strings exactly whenever this record "
            "reports the same measurement. Introduce a new metric only for a "
            "measurement none of them names.\n\n"
            if vocabulary
            else ""
        )
        try:
            response = await self._router.chat(
                [
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{known}"
                            f"RECORD {record.record_id}\n"
                            f"TEXT:\n{record.text}\n\n"
                            "Return the structured profile JSON now."
                        ),
                    },
                ],
                stage="mapper",
                temperature=0.0,
                max_tokens=1600,
                json_mode=True,
                chunk_id=record.record_id,
                question_ids=[],
            )
            if response.usage:
                self._tracker.record(response.usage)
            payload = json_payload(response.content)
        except Exception as exc:
            logger.warning("Profiling failed for %s: %s", record.record_id, exc)
            return None
        if not isinstance(payload, dict):
            return None
        self._save_cache(key, payload)
        return payload

    def _cache_key(
        self,
        record: CompiledRecord,
        model: str,
        vocabulary: list[str],
    ) -> str:
        payload = json.dumps(
            {
                "contract": PROFILER_CONTRACT_VERSION,
                "model": model,
                "prompt": self._prompt_hash,
                "text": hashlib.sha256(record.text.encode()).hexdigest(),
                "vocabulary": sorted(vocabulary),
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
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not write profile cache: %s", exc)


def _needs_profiling(document: CompiledDocument) -> bool:
    """Profile only a repeated-entity registry the layout parser left holed.

    Profiling repairs a fact table, so it is pointless on documents with no
    repeated structure, and wasteful when the deterministic compiler already
    bound a fact card for every record.
    """
    if document.record_kind != "repeated_entity" or not document.records:
        return False
    return any(not record.number_facts for record in document.records)


def _quote_page(record: CompiledRecord, quote: str) -> int:
    """Locate the page whose text contains the quote."""
    wanted = normalize_source(quote)
    markers = list(re.finditer(r"\[Page (\d+)\]\s*", record.text))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(record.text)
        if wanted in normalize_source(record.text[marker.end() : end]):
            return int(marker.group(1))
    return record.page_start


def _discovered_catalog(
    records: list[CompiledRecord],
    minimum_coverage: float,
) -> list[str]:
    """Return metrics repeated widely enough to behave as schema fields.

    A metric only supports aggregation when the document reports it across its
    records, so one-off numbers stay available as evidence but never present
    themselves as a column to count or maximise over.
    """
    if not records:
        return []
    coverage: dict[str, set[str]] = {}
    for record in records:
        for fact in record.number_facts:
            coverage.setdefault(fact.field, set()).add(record.record_id)
    threshold = max(2, round(len(records) * minimum_coverage))
    return sorted(
        field for field, holders in coverage.items() if len(holders) >= threshold
    )


def field_coverage(records: list[CompiledRecord]) -> dict[str, int]:
    """Return how many records report each metric."""
    coverage: dict[str, set[str]] = {}
    for record in records:
        for fact in record.number_facts:
            coverage.setdefault(fact.field, set()).add(record.record_id)
    return {field: len(holders) for field, holders in coverage.items()}
