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
from app.v3.compiler import all_mapping_records, compile_document
from app.v3.exhaustive_mapper import ExhaustiveMapper
from app.v3.models import (
    CompiledDocument,
    ExecutionResult,
    Strategy,
    V3MapResult,
    V3QuestionPlan,
)
from app.v3.question_compiler import compile_questions
from app.v3.reducers import (
    build_evidence_packet,
    reduce_absence,
    reduce_mapped_structure,
    status_counts,
)
from app.v3.structured_executor import execute_structured

logger = get_logger("pipeline.v3")
ProgressCallback = Callable[[str, int, int, int], None]

_PAGE_IN_QUOTE_RE = re.compile(r"\(page\s+(\d+)\)", re.IGNORECASE)


def _page_from_quote_text(quote: str) -> int | None:
    """Recover a page number some evidence strings embed inline (e.g. the
    hardcoded structured-answer helpers), for callers that only get a
    per-question source_pages list rather than a page aligned to each quote.
    """
    match = _PAGE_IN_QUOTE_RE.search(quote)
    return int(match.group(1)) if match else None


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
        if compiled is not None:
            return compiled
        path = self._settings.parsed_dir / f"{document_id}_parsed.json"
        if not path.exists():
            return None
        _, pages = load_parsed_output(path)
        compiled = compile_document(document_id, pages)
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
        compiled = self._load_or_compile(document_id)
        if compiled is None:
            run.warnings.append("No compiled or parsed document found")
            run.completed_at = datetime.now(timezone.utc)
            run.submission_path = str(writer.finalize(run))
            self._store.save_run(run)
            return run

        records = all_mapping_records(compiled)
        expected_record_ids = {record.record_id for record in records}
        plans = compile_questions(questions)
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
    ) -> tuple[ExecutionResult, list[V3MapResult]]:
        if plan.strategy == Strategy.ABSENCE_MATRIX:
            result = reduce_absence(plan, question_results, expected_record_ids)
            if result is None:
                result = reduce_absence(
                    plan,
                    question_results,
                    expected_record_ids,
                    allow_partial=True,
                )
        else:
            packet = build_evidence_packet(plan, question_results, expected_record_ids)
            result = reduce_mapped_structure(plan, packet)

        if result is not None:
            return result, question_results
        packet = build_evidence_packet(plan, question_results, expected_record_ids)
        return await self._answerer.generate(plan, packet), question_results

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
        # evidence_pages is index-aligned with evidence only along the V3
        # generic path (answerer.py / reducers.py); older hardcoded answer
        # helpers don't populate it, so fall back to an unpaired page.
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
        final_answer = result.answer.strip() or "Error: grounded answer unavailable"
        return PipelineAnswer(
            question_id=plan.question_id,
            final_answer=final_answer,
            answer_with_evidence=evidence_note,
            evidence_quotes=quotes,
            source_pages=pages,
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
