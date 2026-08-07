"""V3 adaptive exhaustive pipeline.

The runtime has one execution path: compile the complete document, answer
typed structured questions in Python, map every record for unresolved
questions, then reduce and synthesize from validated evidence.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import PipelineMode, Settings
from app.exceptions import LLMConfigurationError, UnsupportedDocumentError
from app.llm.base import BaseLLMClient
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.parsing.pdf_parser import (
    load_parsed_output,
    parse_pdf,
    save_parsed_output,
)
from app.parsing.text_parser import parse_text
from app.schemas import (
    CoverageReport,
    DocumentChunk,
    DocumentMetadata,
    EvidenceQuote,
    OperationResult,
    Operator,
    PipelineAnswer,
    PipelineRun,
    QuestionRequest,
)
from app.storage.run_store import RunStore
from app.submission import IncrementalSubmissionWriter
from app.v3.answerer import V3Answerer
from app.v3.compiler import COMPILER_VERSION, all_mapping_records, compile_document
from app.v3.exhaustive_mapper import ExhaustiveMapper
from app.v3.field_binder import FieldBinder
from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    EvidencePacket,
    ExecutionResult,
    Strategy,
    V3MapResult,
    V3QuestionPlan,
)
from app.v3.question_compiler import compile_questions
from app.v3.record_profiler import RecordProfiler
from app.v3.reducers import (
    build_evidence_packet,
    reduce_absence,
    reduce_mapped_structure,
    status_counts,
)
from app.v3.shape_classifier import ShapeClassifier
from app.v3.structure_augmenter import StructureAugmenter
from app.v3.structured_executor import execute_structured

logger = get_logger("pipeline.v3")
ProgressCallback = Callable[[str, int, int, int], None]

_PAGE_IN_QUOTE_RE = re.compile(r"\(page\s+(\d+)\)", re.IGNORECASE)


def _page_from_quote_text(quote: str) -> int | None:
    match = _PAGE_IN_QUOTE_RE.search(quote)
    return int(match.group(1)) if match else None

# How many of a question's distinctive words a dismissed record must carry
# before the scan's verdict on it is worth a second reading.
_MIN_DISMISSED_TERMS = 2
# A word carried by more than this share of records cannot single one out.
_MAX_TERM_SHARE = 0.25
_QUESTION_STOPWORDS = frozenset({
    "about", "across", "after", "among", "according", "book", "chapter",
    "does", "each", "from", "give", "guide", "have", "into", "report",
    "state", "that", "their", "these", "they", "this", "what", "when",
    "where", "which", "with", "year", "entries", "entry", "every", "name",
    "many", "much", "over", "same", "some", "there", "those", "were",
})


def _weighted_phrases(
    terms: list[str],
    records: list[CompiledRecord],
) -> list[tuple[str, int]]:
    """Weigh the router's wording by how far it narrows the document.

    "First ascent" appears in two chapters and points at them; "highest"
    appears in most and points almost nowhere. Dropping the common wording
    would lose the only reach some questions have, so it is kept and counted
    for less than the wording that singles a record out.
    """
    folded = [record.text.casefold() for record in records]
    ceiling = max(1, len(folded) * _MAX_TERM_SHARE)
    weighted: list[tuple[str, int]] = []
    for phrase in (term.casefold() for term in terms if len(term) >= 4):
        reach = sum(phrase in text for text in folded)
        if not reach:
            continue
        weighted.append((phrase, _MIN_DISMISSED_TERMS if reach <= ceiling else 1))
    return weighted


def _distinctive_terms(text: str, records: list[CompiledRecord]) -> set[str]:
    """Return the question's words that could locate it within this document.

    A word most records carry says nothing about which record answers the
    question: in a book of parks, "park" is in every chapter. Only the words
    that are rare across the document can single a record out, and which words
    those are is a property of the document rather than of the subject.
    """
    words = {
        term
        for term in re.findall(r"[a-z0-9]+", text.casefold())
        if len(term) >= 4 and term not in _QUESTION_STOPWORDS
    }
    if not records:
        return words
    ceiling = max(1, len(records) * _MAX_TERM_SHARE)
    folded = [record.text.casefold() for record in records]
    return {
        term
        for term in words
        if sum(term in text for text in folded) <= ceiling
    }


class FullScanPipeline:
    """Question-independent compilation plus adaptive exhaustive reduction."""

    def __init__(
        self,
        settings: Settings,
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        self._settings = settings
        warnings = settings.model_policy_warnings()
        for warning in warnings:
            logger.warning("Model policy: %s", warning)
        if settings.evaluation_mode and warnings:
            raise LLMConfigurationError("; ".join(warnings))
        self._router = LLMRouter(settings, client=llm_client)
        self._tracker = UsageTracker()
        self._mapper = ExhaustiveMapper(self._router, self._tracker, settings)
        self._answerer = V3Answerer(self._router, self._tracker)
        self._augmenter = StructureAugmenter(self._router, self._tracker, settings)
        self._profiler = RecordProfiler(self._router, self._tracker, settings)
        self._binder = FieldBinder(self._router, self._tracker, settings)
        self._shaper = ShapeClassifier(self._router, self._tracker, settings)
        self._store = RunStore(settings)

    async def process_document(
        self,
        document_path: Path,
    ) -> tuple[DocumentMetadata, list[DocumentChunk]]:
        """Parse and compile every page into terminal V3 mapping records."""
        extension = document_path.suffix.lower()
        if extension == ".pdf":
            metadata, pages = parse_pdf(document_path)
        elif extension == ".txt":
            metadata, pages = parse_text(document_path)
        else:
            raise UnsupportedDocumentError(
                f"Supported document types are PDF and TXT, got: {extension}"
            )

        save_parsed_output(metadata, pages, self._settings.parsed_dir)
        compiled = compile_document(metadata.document_id, pages)
        compiled = await self._augmenter.augment(compiled)
        compiled = await self._profiler.profile(compiled)
        records = all_mapping_records(compiled)
        chunks = [
            self._record_chunk(metadata.document_id, index, record)
            for index, record in enumerate(records)
        ]
        metadata.chunk_count = len(chunks)

        self._store.save_document_metadata(metadata)
        self._store.save_chunks(metadata.document_id, chunks)
        self._store.save_compiled_document(compiled)
        self._store.save_artifact(
            metadata.document_id,
            "v3_compile_manifest",
            {
                "compiler_version": compiled.compiler_version,
                "record_kind": compiled.record_kind,
                "catalog_records": len(compiled.records),
                "supplementary_records": len(compiled.supplementary_records),
                "mapping_records": len(records),
                "registry_trusted": compiled.registry_trusted,
                "registry_signals": compiled.registry_signals,
                "tables": [
                    {
                        "table_id": table.table_id,
                        "number": table.number,
                        "rows": len(table.rows),
                        "trusted": table.trusted,
                    }
                    for table in compiled.tables
                ],
                "contents_entries": len(compiled.contents_entries),
                "contents_trusted": compiled.contents_trusted,
                "field_catalog": compiled.field_catalog,
                "warnings": compiled.warnings,
            },
        )
        logger.info(
            "V3 document compiled: %d pages, %d catalog records, %d mapping records",
            metadata.page_count,
            len(compiled.records),
            len(records),
        )
        return metadata, chunks

    @staticmethod
    def _record_chunk(document_id: str, index: int, record) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_id=record.record_id,
            chunk_index=index,
            section_id=record.record_id,
            section_title=record.title,
            section_titles=[record.title],
            page_start=record.page_start,
            page_end=record.page_end,
            text=record.text,
        )
        chunk.compute_hash()
        return chunk

    def _load_or_compile(self, document_id: str) -> CompiledDocument | None:
        compiled = self._store.load_compiled_document(document_id)
        if compiled is not None and compiled.compiler_version == COMPILER_VERSION:
            return compiled
        path = self._settings.parsed_dir / f"{document_id}_parsed.json"
        if not path.exists():
            return None
        if compiled is not None:
            logger.info(
                "Recompiling %s because compiler version changed from %s to %s",
                document_id,
                compiled.compiler_version,
                COMPILER_VERSION,
            )
        _, pages = load_parsed_output(path)
        compiled = compile_document(document_id, pages)
        self._store.save_compiled_document(compiled)
        return compiled

    async def _load_or_compile_augmented(
        self,
        document_id: str,
    ) -> CompiledDocument | None:
        compiled = self._load_or_compile(document_id)
        if compiled is None:
            return None
        compiled = await self._augmenter.augment(compiled)
        compiled = await self._profiler.profile(compiled)
        self._store.save_compiled_document(compiled)
        return compiled

    async def answer_questions(
        self,
        document_id: str,
        questions: list[QuestionRequest],
        *,
        mode: Optional[PipelineMode] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineRun:
        """Answer all questions through the sole adaptive-hierarchical V3 route."""
        del mode
        self._tracker = UsageTracker()
        self._mapper = ExhaustiveMapper(self._router, self._tracker, self._settings)
        self._answerer = V3Answerer(self._router, self._tracker)
        self._augmenter = StructureAugmenter(
            self._router,
            self._tracker,
            self._settings,
        )
        self._profiler = RecordProfiler(self._router, self._tracker, self._settings)
        self._binder = FieldBinder(self._router, self._tracker, self._settings)
        self._shaper = ShapeClassifier(self._router, self._tracker, self._settings)
        started = time.perf_counter()
        run = PipelineRun(
            document_id=document_id,
            pipeline_mode="ADAPTIVE_HIERARCHICAL_V3",
            started_at=datetime.now(timezone.utc),
        )
        writer = IncrementalSubmissionWriter(
            self._settings.runs_dir / f"{run.run_id}_submission.json",
            team=self._settings.team_name,
            notes=self._settings.submission_notes,
        )
        run.submission_path = str(writer.initialize(questions))
        compiled = await self._load_or_compile_augmented(document_id)
        if compiled is None:
            run.warnings.append("No compiled or parsed document found")
            run.completed_at = datetime.now(timezone.utc)
            run.submission_path = str(writer.finalize(run))
            self._store.save_run(run)
            return run

        records = all_mapping_records(compiled)
        expected_record_ids = {record.record_id for record in records}
        shapes = await self._shaper.classify(questions, compiled)
        bindings = await self._binder.bind(questions, compiled)
        plans = compile_questions(questions, compiled, bindings, shapes)
        deterministic: dict[str, ExecutionResult] = {}
        unresolved: list[V3QuestionPlan] = []
        if progress_callback:
            progress_callback("compiling", 0, len(plans), 0)
        for index, plan in enumerate(plans, start=1):
            result = execute_structured(plan, compiled)
            if result is None:
                unresolved.append(plan)
            else:
                deterministic[plan.question_id] = result
            if progress_callback:
                progress_callback("compiling", index, len(plans), 0)

        map_results = await self._mapper.map_document(
            compiled,
            unresolved,
            progress_callback=progress_callback,
        )
        callback = None
        if progress_callback:

            def repair_progress(
                _stage: str,
                done: int,
                total: int,
                failed: int,
            ) -> None:
                progress_callback("repairing", done, total, failed)

            callback = repair_progress
        map_results = await self._mapper.repair_failures(
            compiled,
            unresolved,
            map_results,
            progress_callback=callback,
            model_override=self._settings.answer_model,
        )
        self._store.save_artifact(
            run.run_id,
            "v3_question_plans",
            [plan.model_dump(mode="json") for plan in plans],
        )

        for index, plan in enumerate(plans, start=1):
            try:
                result = deterministic.get(plan.question_id)
                question_results = [
                    item for item in map_results if item.question_id == plan.question_id
                ]
                if result is None:
                    result, question_results = await self._reduce_question(
                        plan,
                        question_results,
                        expected_record_ids,
                        compiled,
                    )
                answer = self._pipeline_answer(
                    plan,
                    result,
                    question_results,
                    compiled,
                )
            except Exception as exc:
                logger.exception("V3 question %s failed", plan.question_id)
                answer = PipelineAnswer(
                    question_id=plan.question_id,
                    final_answer=f"Error: {exc}",
                    operator=self._operator(plan),
                    warnings=[str(exc)],
                )
            run.answers.append(answer)
            writer.update(answer)
            if progress_callback:
                progress_callback("answering", index, len(plans), 0)

        self._store.save_artifact(
            run.run_id,
            "v3_map_results",
            [result.model_dump(mode="json") for result in map_results],
        )
        run.completed_at = datetime.now(timezone.utc)
        run.usage = self._tracker.records
        run.total_input_tokens = self._tracker.total_input_tokens
        run.total_output_tokens = self._tracker.total_output_tokens
        run.total_latency_ms = (time.perf_counter() - started) * 1000
        run.submission_path = str(writer.finalize(run))
        self._store.save_run(run)
        return run

    async def _reduce_question(
        self,
        plan: V3QuestionPlan,
        question_results: list[V3MapResult],
        expected_record_ids: set[str],
        compiled: CompiledDocument,
    ) -> tuple[ExecutionResult, list[V3MapResult]]:
        result = self._reduced_result(
            plan,
            question_results,
            expected_record_ids,
            compiled,
        )

        if result is not None:
            return result, question_results
        packet = build_evidence_packet(plan, question_results, expected_record_ids)
        dismissed = self._dismissed_matches(plan, compiled, question_results)
        should_repair = self._evidence_needs_repair(plan, packet) or bool(dismissed)
        if should_repair:
            # Both selections answer different questions — which records the
            # scan dismissed, and which records the question's words point at
            # — so revisiting only one of them narrows what repair can find.
            selected = list(
                {
                    record.record_id: record
                    for record in [*dismissed, *self._lexical_repair_records(plan, compiled)]
                }.values()
            )[:8]
            repaired = await self._mapper.map_selected_records(
                compiled,
                selected,
                [plan],
                model_override=self._settings.answer_model,
            )
            if repaired:
                replacement = {
                    (item.question_id, item.record_id): item for item in repaired
                }
                question_results = [
                    self._merge_map_result(
                        item,
                        replacement.get((item.question_id, item.record_id)),
                    )
                    for item in question_results
                ]
                existing = {
                    (item.question_id, item.record_id) for item in question_results
                }
                question_results.extend(
                    item
                    for item in repaired
                    if (item.question_id, item.record_id) not in existing
                )
                result = self._reduced_result(
                    plan,
                    question_results,
                    expected_record_ids,
                    compiled,
                )
                if result is not None:
                    return result, question_results
                packet = build_evidence_packet(
                    plan,
                    question_results,
                    expected_record_ids,
                )
        return await self._answerer.generate(plan, packet), question_results

    @staticmethod
    def _dismissed_matches(
        plan: V3QuestionPlan,
        compiled: CompiledDocument,
        results: list[V3MapResult],
        *,
        limit: int = 4,
    ) -> list[CompiledRecord]:
        """Return records the scan dismissed although they answer its terms.

        A question that needs several pieces of evidence is only as good as
        the records the scan admits, and a small model reading a long record
        misses figures it plainly states. Where a cheap word match and the
        scan's verdict disagree — the record carries the question's own
        distinctive terms, yet came back with nothing — the verdict is the
        side more likely to be wrong, so those records are read again by the
        stronger model. Every record is still scanned first; this only revisits.
        """
        if plan.strategy == Strategy.ABSENCE_MATRIX:
            return []
        empty = {
            result.record_id
            for result in results
            if result.question_id == plan.question_id and result.status == "no_evidence"
        }
        if not empty:
            return []
        records = all_mapping_records(compiled)
        wanted = _distinctive_terms(plan.question, records)
        # The router read the question and offered the wording a document
        # would use for what it asks about. A chapter that says "crosses into
        # Kaliningrad" shares no rare word with a question about international
        # borders, and only that wording can reach it.
        phrases = _weighted_phrases(plan.shape_plan.search_terms, records)
        if len(wanted) < 2 and not phrases:
            return []
        scored: list[tuple[int, CompiledRecord]] = []
        for record in records:
            if record.record_id not in empty:
                continue
            folded = record.text.casefold()
            matched = sum(term in folded for term in wanted)
            matched += sum(weight for phrase, weight in phrases if phrase in folded)
            if matched >= _MIN_DISMISSED_TERMS:
                scored.append((matched, record))
        scored.sort(key=lambda item: (-item[0], item[1].ordinal))
        return [record for _, record in scored[:limit]]

    @staticmethod
    def _evidence_needs_repair(plan: V3QuestionPlan, packet: EvidencePacket) -> bool:
        if not packet.evidence:
            return True
        if plan.category != "global_synthesis":
            return False
        record_ids = {item.record_id for item in packet.evidence}
        expected = max(packet.expected_records, 1)
        positions = {
            min(2, int(3 * max(item.record_ordinal - 1, 0) / expected))
            for item in packet.evidence
        }
        return len(record_ids) < 6 or len(positions) < 3

    @staticmethod
    def _merge_map_result(
        original: V3MapResult,
        repaired: V3MapResult | None,
    ) -> V3MapResult:
        if repaired is None:
            return original
        evidence = list(original.evidence)
        seen_evidence = {
            (item.exact_quote.casefold(), item.page, item.role) for item in evidence
        }
        for item in repaired.evidence:
            key = (item.exact_quote.casefold(), item.page, item.role)
            if key not in seen_evidence:
                evidence.append(item)
                seen_evidence.add(key)

        topic_priority = {"none": 0, "uncertain": 1, "mention_only": 2, "substantive": 3}
        topics = {item.topic.casefold(): item for item in original.topics}
        for item in repaired.topics:
            key = item.topic.casefold()
            previous = topics.get(key)
            if previous is None or topic_priority.get(item.level, 0) > topic_priority.get(
                previous.level,
                0,
            ):
                topics[key] = item
        status = "evidence_found" if evidence else repaired.status
        return V3MapResult(
            question_id=original.question_id,
            record_id=original.record_id,
            status=status,
            evidence=evidence,
            topics=list(topics.values()),
            error="" if evidence or topics else repaired.error or original.error,
        )

    @staticmethod
    def _reduced_result(
        plan: V3QuestionPlan,
        question_results: list[V3MapResult],
        expected_record_ids: set[str],
        compiled: CompiledDocument | None = None,
    ) -> ExecutionResult | None:
        if plan.strategy == Strategy.ABSENCE_MATRIX:
            result = reduce_absence(
                plan,
                question_results,
                expected_record_ids,
                document=compiled,
            )
            return result or reduce_absence(
                plan,
                question_results,
                expected_record_ids,
                allow_partial=True,
                document=compiled,
            )
        packet = build_evidence_packet(plan, question_results, expected_record_ids)
        return reduce_mapped_structure(plan, packet, compiled)

    @staticmethod
    def _lexical_repair_records(
        plan: V3QuestionPlan,
        compiled: CompiledDocument,
        *,
        limit: int = 8,
    ) -> list[CompiledRecord]:
        records = all_mapping_records(compiled)
        stop = {
            "about", "across", "after", "among", "according", "book", "chapter",
            "does", "each", "from", "give", "guide", "have", "into", "report",
            "state", "that", "their", "these", "they", "this", "what", "when",
            "where", "which", "with", "year",
        }

        def terms(text: str) -> set[str]:
            return {
                term
                for term in re.findall(r"[a-z0-9]+", text.casefold())
                if len(term) >= 4 and term not in stop
            }

        question_terms = terms(plan.question)

        def score(record, wanted: set[str]) -> tuple[int, int]:
            folded = record.text.casefold()
            matched = sum(term in folded for term in wanted)
            hint_score = 20 * sum(
                hint.casefold() in folded for hint in plan.entity_hints
            )
            return hint_score + matched, -record.ordinal

        selected: dict[str, CompiledRecord] = {}
        if plan.strategy == Strategy.ABSENCE_MATRIX:
            for topic in plan.candidate_topics:
                wanted = terms(topic)
                ranked = sorted(records, key=lambda record: score(record, wanted), reverse=True)
                for record in ranked[:2]:
                    if score(record, wanted)[0] > 0:
                        selected[record.record_id] = record
        if plan.category == "global_synthesis":
            maximum_ordinal = max((record.ordinal for record in records), default=1)
            buckets: list[list[CompiledRecord]] = [[], [], []]
            for record in records:
                position = min(2, int(3 * max(record.ordinal - 1, 0) / maximum_ordinal))
                buckets[position].append(record)
            quotas = [3, 3, 2]
            for bucket, quota in zip(buckets, quotas, strict=True):
                ranked_bucket = sorted(
                    bucket,
                    key=lambda record: score(record, question_terms),
                    reverse=True,
                )
                for record in ranked_bucket[:quota]:
                    if score(record, question_terms)[0] > 0:
                        selected[record.record_id] = record
        ranked = sorted(records, key=lambda record: score(record, question_terms), reverse=True)
        for record in ranked:
            if len(selected) >= limit:
                break
            if score(record, question_terms)[0] > 0:
                selected[record.record_id] = record
        return list(selected.values())[:limit]

    def _pipeline_answer(
        self,
        plan: V3QuestionPlan,
        result: ExecutionResult,
        map_results: list[V3MapResult],
        compiled: CompiledDocument,
    ) -> PipelineAnswer:
        operator = self._operator(plan)
        pages = sorted(set(result.source_pages))
        evidence_note = (
            "Validated source pages/segments: " + ", ".join(map(str, pages)) if pages else ""
        )
        final_answer = result.answer.strip() or (
            "The available document evidence was insufficient to determine a more "
            "specific answer."
        )
        paired_pages = (
            result.evidence_pages
            if len(result.evidence_pages) == len(result.evidence)
            else [None] * len(result.evidence)
        )
        seen_quotes: set[str] = set()
        quotes: list[EvidenceQuote] = []
        for quote, page in zip(result.evidence, paired_pages):
            if quote and quote not in seen_quotes:
                seen_quotes.add(quote)
                quotes.append(EvidenceQuote(quote=quote, page=page or _page_from_quote_text(quote)))
        return PipelineAnswer(
            question_id=plan.question_id,
            final_answer=final_answer,
            answer_with_evidence=evidence_note,
            operator=operator,
            operation_result=OperationResult(
                question_id=plan.question_id,
                operator=operator,
                result_value=result.answer or None,
                computation_detail=f"V3 strategy: {plan.strategy.value}",
                source_pages=pages,
                warnings=result.warnings,
            ),
            coverage=self._coverage(plan, map_results, compiled),
            warnings=result.warnings,
            evidence_quotes=quotes,
            source_pages=pages,
        )

    @staticmethod
    def _operator(plan: V3QuestionPlan) -> Operator:
        if plan.strategy == Strategy.ABSENCE_MATRIX:
            return Operator.ABSENCE
        if plan.strategy == Strategy.CLAIM_COMPARE:
            return Operator.COMPARE
        if plan.strategy == Strategy.HIERARCHICAL_SYNTHESIS:
            return (
                Operator.GENERAL_SYNTHESIS
                if plan.category == "global_synthesis"
                else Operator.MULTI_HOP
            )
        if plan.strategy == Strategy.EXHAUSTIVE_LOOKUP:
            return Operator.LOOKUP
        operation = str(plan.metadata.get("operation", ""))
        return Operator.ARGMAX if operation == "argmax" else Operator.FILTER_COUNT_LIST

    @staticmethod
    def _coverage(
        plan: V3QuestionPlan,
        results: list[V3MapResult],
        compiled: CompiledDocument,
    ) -> CoverageReport:
        total = len(all_mapping_records(compiled))
        if not results:
            return CoverageReport(
                total_pages=compiled.page_count,
                parsed_pages=compiled.page_count,
                total_chunks=total,
                successful_mappings=total,
                mapped_chunks=total,
                coverage_percentage=100.0,
                required_percentage=100.0,
                requirement_met=True,
            )
        counts = status_counts(plan.question_id, results)
        successful = counts["evidence_found"] + counts["no_evidence"]
        failed = counts["parse_failed"] + counts["llm_failed"]
        percentage = round(successful / total * 100, 2) if total else 100.0
        return CoverageReport(
            total_pages=compiled.page_count,
            parsed_pages=compiled.page_count,
            total_chunks=total,
            successful_mappings=successful,
            no_evidence_chunks=counts["no_evidence"],
            uncertain_chunks=counts["uncertain"],
            failed_chunks=failed,
            coverage_percentage=percentage,
            required_percentage=100.0,
            requirement_met=successful == total,
            mapped_chunks=len(results),
            missing_chunks=max(0, total - len(results)),
        )

    async def close(self) -> None:
        await self._router.close()
