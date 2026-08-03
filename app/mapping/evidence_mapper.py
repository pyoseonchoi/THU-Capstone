"""Single-chunk evidence mapper.

Sends one chunk + question batch to the mapper LLM and parses
structured evidence from the response.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from app.field_names import canonical_field_name
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.reduction.normalizer import normalize_extracted_value
from app.schemas import (
    ChunkMapResult,
    DocumentChunk,
    EvidenceItem,
    ExtractionStatus,
    QueryPlan,
    TopicAssessment,
)

logger = get_logger("mapping.evidence_mapper")

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "mapper_system.txt"


def _build_cache_key(
    chunk: DocumentChunk,
    plans: list[QueryPlan],
    model: str,
    prompt_version: str,
) -> str:
    """Build a cache key from the complete mapper input contract."""
    payload = {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "content_hash": chunk.content_hash,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "model": model,
        "prompt_version": prompt_version,
        "plans": [
            plan.model_dump(mode="json")
            for plan in sorted(plans, key=lambda item: item.question_id)
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _as_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_string(item) for item in value if _as_string(item)]
    text = _as_string(value)
    return [text] if text else []


def _bounded_float(value: object, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _bounded_page(value: object, chunk: DocumentChunk, default: int) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return default
    return min(chunk.page_end, max(chunk.page_start, page))


class EvidenceMapper:
    """Maps a single chunk against a batch of questions."""

    def __init__(
        self,
        router: LLMRouter,
        usage_tracker: UsageTracker,
        cache_dir: Optional[Path] = None,
        prompt_version: str = "v1",
    ):
        self._router = router
        self._tracker = usage_tracker
        self._system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
        self._cache_dir = cache_dir
        self._prompt_version = prompt_version
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    async def map_chunk(
        self,
        chunk: DocumentChunk,
        plans: list[QueryPlan],
        *,
        model_override: str | None = None,
    ) -> list[ChunkMapResult]:
        """Extract evidence from one chunk for the given questions.

        Returns one ChunkMapResult per question.
        """
        model = model_override or self._router.get_model("mapper")
        prompt_version = self._prompt_version

        # Check cache
        cache_key = _build_cache_key(chunk, plans, model, prompt_version)
        cached = self._load_cache(cache_key, chunk, plans)
        if cached is not None:
            logger.debug("Cache hit for chunk %s", chunk.chunk_id)
            return cached

        # Build the user message
        questions_desc = "\n".join(
            json.dumps(p.model_dump(mode="json"), ensure_ascii=False)
            for p in plans
        )

        user_msg = (
            f"CHUNK ID: {chunk.chunk_id}\n"
            f"SECTIONS: {chunk.section_titles or [chunk.section_title]}\n"
            f"PAGES: {chunk.page_start}-{chunk.page_end}\n\n"
            f"CHUNK TEXT:\n{chunk.text}\n\n"
            f"QUESTIONS:\n{questions_desc}\n\n"
            f"Extract evidence for each question. Return JSON with "
            f'{{"results": [<one result per question>]}}'
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

        question_ids = [p.question_id for p in plans]

        try:
            response = await self._router.chat(
                messages,
                stage="mapper",
                json_mode=True,
                max_tokens=min(8192, max(4096, 1024 * len(plans))),
                chunk_id=chunk.chunk_id,
                question_ids=question_ids,
                model_override=model_override,
            )
            if response.usage:
                self._tracker.record(response.usage)

            results = self._parse_response(response.content, chunk, plans)
        except Exception as exc:
            logger.error(
                "Mapper failed for chunk %s: %s", chunk.chunk_id, exc
            )
            # Return failed results for all questions
            results = [
                ChunkMapResult(
                    chunk_id=chunk.chunk_id,
                    question_id=p.question_id,
                    extraction_status=ExtractionStatus.LLM_FAILED,
                    uncertainty_notes=str(exc),
                )
                for p in plans
            ]

        if results and all(
            result.extraction_status in (
                ExtractionStatus.EVIDENCE_FOUND,
                ExtractionStatus.NO_EVIDENCE,
            )
            for result in results
        ):
            self._save_cache(cache_key, results)
        return results

    def _parse_response(
        self,
        raw: str,
        chunk: DocumentChunk,
        plans: list[QueryPlan],
    ) -> list[ChunkMapResult]:
        """Parse the mapper LLM response into ChunkMapResults."""
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
        except Exception:
            logger.warning("Mapper returned invalid JSON for chunk %s", chunk.chunk_id)
            return [
                ChunkMapResult(
                    chunk_id=chunk.chunk_id,
                    question_id=p.question_id,
                    extraction_status=ExtractionStatus.PARSE_FAILED,
                    uncertainty_notes="Invalid JSON response",
                )
                for p in plans
            ]

        if isinstance(data, list):
            raw_results = data
        elif isinstance(data, dict):
            raw_results = data.get("results", [])
        else:
            raw_results = []
        if not isinstance(raw_results, list):
            raw_results = []

        # Map results to questions
        plan_map = {p.question_id: p for p in plans}
        result_by_qid: dict[str, ChunkMapResult] = {}

        for r in raw_results:
            if not isinstance(r, dict):
                continue
            qid = r.get("question_id", "")
            if qid not in plan_map:
                continue
            status_str = _as_string(
                r.get("extraction_status", "no_evidence")
            ).lower()
            try:
                status = ExtractionStatus(status_str)
            except ValueError:
                status = ExtractionStatus.UNCERTAIN

            evidence_items: list[EvidenceItem] = []
            raw_evidence = r.get("evidence_items", [])
            if not isinstance(raw_evidence, list):
                raw_evidence = []
            for ei in raw_evidence:
                if not isinstance(ei, dict):
                    continue
                try:
                    field_name = canonical_field_name(
                        _as_string(ei.get("field_name", ""))
                    )
                    plan = plan_map[qid]
                    extraction_field = next(
                        (
                            field
                            for field in plan.extraction_fields
                            if field.field_name == field_name
                        ),
                        None,
                    )
                    raw_value = _as_string(ei.get("raw_value", ""))
                    exact_quote = _as_string(ei.get("exact_quote", ""))
                    unit = _as_string(ei.get("unit", ""))
                    normalized_value = normalize_extracted_value(
                        raw_value,
                        ei.get("normalized_value"),
                        expected_type=(
                            extraction_field.expected_type
                            if extraction_field is not None
                            else ""
                        ),
                        from_unit=unit,
                        target_unit=plan.normalized_unit,
                        supporting_text=exact_quote,
                    )
                    item_page_start = _bounded_page(
                        ei.get("page_start", ei.get("page")),
                        chunk,
                        chunk.page_start,
                    )
                    item_page_end = _bounded_page(
                        ei.get("page_end", ei.get("page")),
                        chunk,
                        item_page_start,
                    )
                    item = EvidenceItem(
                        question_id=qid,
                        chunk_id=chunk.chunk_id,
                        section_id=chunk.section_id,
                        page_start=min(item_page_start, item_page_end),
                        page_end=max(item_page_start, item_page_end),
                        entity_name=_as_string(ei.get("entity_name", "")),
                        entity_id=_as_string(ei.get("entity_id", "")),
                        field_name=field_name,
                        raw_value=raw_value,
                        normalized_value=normalized_value,
                        unit=unit,
                        claim=_as_string(ei.get("claim", "")),
                        exact_quote=exact_quote,
                        relevance=_bounded_float(ei.get("relevance", 0.5)),
                        confidence=_bounded_float(ei.get("confidence", 0.5)),
                        uncertainty=_as_string(ei.get("uncertainty", "")),
                        extraction_status=status,
                    )
                    evidence_items.append(item)
                except (ValueError, TypeError) as exc:
                    logger.warning("Skipping malformed evidence item: %s", exc)

            topic_assessments: list[TopicAssessment] = []
            raw_assessments = r.get("topic_assessments", [])
            if not isinstance(raw_assessments, list):
                raw_assessments = []
            for ta in raw_assessments:
                if not isinstance(ta, dict):
                    continue
                try:
                    topic_assessments.append(TopicAssessment(
                        topic=_as_string(ta.get("topic", "")),
                        level=_as_string(ta.get("level", "uncertain")).lower(),
                        justification=_as_string(ta.get("justification", "")),
                    ))
                except (ValueError, TypeError):
                    pass

            if status == ExtractionStatus.EVIDENCE_FOUND and not evidence_items:
                status = ExtractionStatus.UNCERTAIN

            result_by_qid[qid] = ChunkMapResult(
                chunk_id=chunk.chunk_id,
                question_id=qid,
                extraction_status=status,
                evidence_items=evidence_items,
                topic_assessments=topic_assessments,
                cross_references=_as_string_list(r.get("cross_references", [])),
                conflicts=_as_string_list(r.get("conflicts", [])),
                uncertainty_notes=_as_string(r.get("uncertainty_notes", "")),
            )

        results: list[ChunkMapResult] = []
        for p in plans:
            if p.question_id in result_by_qid:
                results.append(result_by_qid[p.question_id])
            else:
                results.append(ChunkMapResult(
                    chunk_id=chunk.chunk_id,
                    question_id=p.question_id,
                    extraction_status=ExtractionStatus.PARSE_FAILED,
                    uncertainty_notes="No result returned for this question",
                ))

        return results

    def _load_cache(
        self,
        key: str,
        chunk: DocumentChunk,
        plans: list[QueryPlan],
    ) -> Optional[list[ChunkMapResult]]:
        """Load cached results if available."""
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached = [ChunkMapResult(**r) for r in data]
            by_question = {result.question_id: result for result in cached}
            if set(by_question) != {plan.question_id for plan in plans}:
                return None
            results: list[ChunkMapResult] = []
            for plan in plans:
                result = by_question[plan.question_id].model_copy(deep=True)
                result.chunk_id = chunk.chunk_id
                for item in result.evidence_items:
                    item.chunk_id = chunk.chunk_id
                    item.section_id = chunk.section_id
                    item.page_start = min(
                        chunk.page_end, max(chunk.page_start, item.page_start)
                    )
                    item.page_end = min(
                        chunk.page_end, max(item.page_start, item.page_end)
                    )
                results.append(result)
            return results
        except Exception:
            return None

    def _save_cache(self, key: str, results: list[ChunkMapResult]) -> None:
        """Save results to cache."""
        if not self._cache_dir:
            return
        path = self._cache_dir / f"{key}.json"
        try:
            data = [r.model_dump(mode="json") for r in results]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data), encoding="utf-8")
            temporary.replace(path)
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)
