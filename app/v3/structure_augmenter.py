"""Bounded LLM augmentation for layouts that plain text cannot disambiguate."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from app.config import Settings
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.v3.compiler import _looks_like_entity_title
from app.v3.models import CompiledDocument, ContentsEntry

logger = get_logger("v3.structure_augmenter")
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_contents_system.txt"
TITLE_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "v3_title_system.txt"
_GROUPS = ("boxes", "spotlights", "figures", "tables")
_AUGMENTER_VERSION = "v3"
_TITLE_AUGMENTER_VERSION = "v1"
_TITLE_EXCERPT_CHARS = 2000


def _json_object(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _identifier_exists(identifier: str, text: str) -> bool:
    return bool(re.search(
        rf"(?<![A-Za-z0-9]){re.escape(identifier)}(?![A-Za-z0-9])",
        text,
        re.I,
    ))


class StructureAugmenter:
    """Compile only ambiguous, bounded structural views and cache the result."""

    def __init__(
        self,
        router: LLMRouter,
        tracker: UsageTracker,
        settings: Settings,
    ) -> None:
        self._router = router
        self._tracker = tracker
        self._settings = settings
        self._prompt = PROMPT_FILE.read_text(encoding="utf-8")
        self._title_prompt = TITLE_PROMPT_FILE.read_text(encoding="utf-8")
        self._cache_dir = settings.cache_dir / "v3" / "structure"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._title_cache_dir = settings.cache_dir / "v3" / "titles"
        self._title_cache_dir.mkdir(parents=True, exist_ok=True)

    def _needed(self, document: CompiledDocument) -> bool:
        compact = re.sub(r"\s+", "", document.contents_text).casefold()
        signals = sum(group in compact for group in _GROUPS)
        return bool(document.contents_text and signals >= 2)

    def _cache_path(self, document: CompiledDocument) -> Path:
        model = self._router.get_model("planner")
        payload = "\n".join((
            _AUGMENTER_VERSION,
            document.document_id,
            document.compiler_version,
            model,
            hashlib.sha256(self._prompt.encode("utf-8")).hexdigest(),
            hashlib.sha256(document.contents_text.encode("utf-8")).hexdigest(),
        ))
        return self._cache_dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    def _validated_entries(
        self,
        payload: dict,
        source: str,
        *,
        minimum_entries: int = 0,
        minimum_by_group: dict[str, int] | None = None,
    ) -> tuple[list[ContentsEntry], bool]:
        entries: list[ContentsEntry] = []
        for group in _GROUPS:
            raw_items = payload.get(group, [])
            if not isinstance(raw_items, list):
                continue
            seen: set[str] = set()
            for raw_item in raw_items:
                identifier = str(raw_item).strip()
                key = identifier.casefold()
                if not identifier or key in seen:
                    continue
                seen.add(key)
                if not _identifier_exists(identifier, source):
                    continue
                entries.append(ContentsEntry(category=group, identifier=identifier))
        present_groups = {entry.category for entry in entries}
        compact = re.sub(r"\s+", "", source).casefold()
        required_groups = {group for group in _GROUPS if group in compact}
        counts = Counter(entry.category for entry in entries)
        group_minimums = minimum_by_group or {}
        trusted = (
            required_groups <= present_groups
            and len(entries) >= minimum_entries
            and all(counts[group] >= minimum for group, minimum in group_minimums.items())
        )
        return entries, trusted

    async def _call(
        self,
        source: str,
        instruction: str = "",
        *,
        system_prompt: str | None = None,
    ) -> dict:
        user_content = source
        if instruction:
            user_content = f"{instruction}\n\nSOURCE:\n{source}"
        response = await self._router.chat(
            [
                {"role": "system", "content": system_prompt or self._prompt},
                {"role": "user", "content": user_content},
            ],
            stage="planner",
            temperature=0.0,
            max_tokens=4000,
            json_mode=True,
            question_ids=[],
        )
        if response.usage:
            self._tracker.record(response.usage)
        return _json_object(response.content)

    async def _grouped_retry(self, source: str) -> dict[str, list[str]]:
        async def extract(group: str) -> tuple[str, list[str]]:
            system_prompt = (
                "You reconstruct one named list from a multi-column publication Contents "
                "page whose text reading order may interleave adjacent columns. Use the "
                "list heading and entry titles to distinguish the requested list from "
                "neighbouring lists. Return every identifier in the requested list, "
                "including O- and S-prefixed identifiers. Scan through the end of the "
                "source, deduplicate, and do not return page numbers or identifiers from "
                "other lists. Return JSON only: {\"identifiers\": [\"1.1\", \"S1.2.1\"]}."
            )
            payload = await self._call(
                source,
                (
                    f"Requested list: {group.upper()}. Extract that list only."
                ),
                system_prompt=system_prompt,
            )
            raw_items = payload.get("identifiers", payload.get(group, []))
            values = raw_items if isinstance(raw_items, list) else []
            return group, [str(item).strip() for item in values if str(item).strip()]

        extracted = await asyncio.gather(*(extract(group) for group in _GROUPS))
        return {group: values for group, values in extracted}

    def _title_cache_path(self, document: CompiledDocument, record) -> Path:
        model = self._router.get_model("planner")
        payload = "\n".join((
            _TITLE_AUGMENTER_VERSION,
            document.document_id,
            record.record_id,
            model,
            hashlib.sha256(self._title_prompt.encode("utf-8")).hexdigest(),
            hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
        ))
        return self._title_cache_dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    async def _resolve_one_title(self, document: CompiledDocument, record) -> None:
        excerpt = record.text[:_TITLE_EXCERPT_CHARS]
        cache_path = self._title_cache_path(document, record)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                title = str(cached.get("title", "")).strip()
            except (OSError, json.JSONDecodeError):
                title = ""
            if title and _identifier_exists(title, excerpt):
                record.title = title
                return

        try:
            response = await self._router.chat(
                [
                    {"role": "system", "content": self._title_prompt},
                    {"role": "user", "content": excerpt},
                ],
                stage="planner",
                temperature=0.0,
                max_tokens=200,
                json_mode=True,
                question_ids=[],
            )
            if response.usage:
                self._tracker.record(response.usage)
            title = str(_json_object(response.content).get("title", "")).strip()
        except Exception as exc:
            logger.warning("Title resolution failed for %s: %s", record.record_id, exc)
            return

        if not title or not _identifier_exists(title, excerpt):
            return
        cache_path.write_text(
            json.dumps({"title": title}, ensure_ascii=False), encoding="utf-8"
        )
        record.title = title

    async def resolve_uncertain_titles(self, document: CompiledDocument) -> CompiledDocument:
        """Re-resolve titles a layout heuristic could not confidently name.

        `compile_document` picks a title from nearby text using layout
        signals (headings, rarity, position). Those signals can be fooled by
        a document whose real chapter title is not heading-formatted while a
        sibling subsection (a hotel listing, a trail name) is -- the
        heuristic then has no way to tell them apart. Rather than add more
        layout rules, hand the ambiguous excerpt to an LLM, which can read it
        the way a person would: the real chapter title reads as an entity
        name, not as a hotel ad or an activity blurb. Every returned title
        must be a literal substring of the excerpt it was read from, so a
        hallucinated name is discarded rather than trusted.
        """
        if document.record_kind != "repeated_entity":
            return document
        uncertain = [
            record for record in document.records
            if not _looks_like_entity_title(record.title)
        ]
        if not uncertain:
            return document
        await asyncio.gather(
            *(self._resolve_one_title(document, record) for record in uncertain)
        )
        unique_titles = len({record.title.casefold() for record in document.records})
        toc_count = document.registry_signals.get("contents_entries", 0)
        document.registry_signals["unique_titles"] = unique_titles
        document.registry_trusted = (
            unique_titles == len(document.records)
            and (not toc_count or toc_count == len(document.records))
            and not any(record.title.startswith("Record ") for record in document.records)
        )
        return document

    async def augment(self, document: CompiledDocument) -> CompiledDocument:
        document = await self.resolve_uncertain_titles(document)
        if document.contents_trusted or not self._needed(document):
            return document
        minimum_entries = max(4, len(document.contents_entries))
        baseline_counts = Counter(entry.category for entry in document.contents_entries)
        minimum_by_group = {
            group: min(count, 4) for group, count in baseline_counts.items()
        }
        cache_path = self._cache_path(document)
        cached_incomplete: dict | None = None
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                entries, trusted = self._validated_entries(
                    payload,
                    document.contents_text,
                    minimum_entries=minimum_entries,
                    minimum_by_group=minimum_by_group,
                )
                if trusted:
                    document.contents_entries = entries
                    document.contents_trusted = True
                    return document
                cached_incomplete = payload
            except (OSError, json.JSONDecodeError):
                pass

        try:
            payload = cached_incomplete or await self._call(document.contents_text)
            entries, trusted = self._validated_entries(
                payload,
                document.contents_text,
                minimum_entries=minimum_entries,
                minimum_by_group=minimum_by_group,
            )
            if not trusted:
                payload = await self._grouped_retry(document.contents_text)
                entries, trusted = self._validated_entries(
                    payload,
                    document.contents_text,
                    minimum_entries=minimum_entries,
                    minimum_by_group=minimum_by_group,
                )
            if trusted:
                document.contents_entries = entries
                document.contents_trusted = True
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                document.warnings.append("Contents index failed source validation")
        except Exception as exc:
            logger.warning("Contents augmentation failed: %s", exc)
            document.warnings.append("Contents index augmentation failed")
        return document
