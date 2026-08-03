"""Main pipeline orchestrator.

Connects all stages: parsing → chunking → planning → mapping →
normalization → reduction → answer generation → verification.

Supports FULLSCAN_OPERATOR (main) and baseline modes.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.answering.answer_generator import AnswerGenerator
from app.chunking.structural_chunker import chunk_document, generate_chunk_manifest
from app.config import PipelineMode, Settings
from app.exceptions import LLMConfigurationError, UnsupportedDocumentError
from app.llm.base import BaseLLMClient
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.logging_config import get_logger
from app.mapping.batch_mapper import BatchMapper
from app.mapping.evidence_mapper import EvidenceMapper
from app.parsing.pdf_parser import parse_pdf, save_parsed_output
from app.parsing.structure_detector import detect_sections
from app.parsing.text_parser import parse_text
from app.planning.question_planner import QuestionPlanner
from app.reduction.deduplicator import deduplicate_entities
from app.reduction.deterministic_reducer import reduce
from app.reduction.evidence_ledger import build_ledger
from app.schemas import (
    DocumentChunk,
    DocumentMetadata,
    PipelineAnswer,
    PipelineRun,
    QuestionRequest,
)
from app.storage.run_store import RunStore
from app.submission import IncrementalSubmissionWriter
from app.verification.claim_verifier import ClaimVerifier
from app.verification.coverage_verifier import verify_coverage

logger = get_logger("pipeline")


class FullScanPipeline:
    """Main FULLSCAN-QA pipeline orchestrator."""

    def __init__(
        self,
        settings: Settings,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self._settings = settings
        policy_warnings = settings.model_policy_warnings()
        for warning in policy_warnings:
            logger.warning("Model policy: %s", warning)
        if settings.evaluation_mode and policy_warnings:
            raise LLMConfigurationError("; ".join(policy_warnings))
        self._usage_tracker = UsageTracker()

        self._router = LLMRouter(settings, client=llm_client)
        self._planner = QuestionPlanner(self._router, self._usage_tracker)
        self._mapper = EvidenceMapper(
            self._router, self._usage_tracker,
            cache_dir=settings.cache_dir,
            prompt_version=settings.prompt_version_mapper,
        )
        self._batch_mapper = BatchMapper(self._mapper, settings)
        self._answer_gen = AnswerGenerator(self._router, self._usage_tracker)
        self._verifier = ClaimVerifier(self._router, self._usage_tracker)
        self._store = RunStore(settings)

    async def process_document(
        self, document_path: Path
    ) -> tuple[DocumentMetadata, list[DocumentChunk]]:
        """Parse and chunk a supported PDF or TXT document.

        Returns:
            Tuple of (metadata, chunks).
        """
        logger.info("Parsing document: %s", document_path)
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

        sections = detect_sections(pages)
        chunks = chunk_document(
            metadata.document_id,
            pages,
            sections,
            target_tokens=self._settings.chunk_target_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
        )
        metadata.chunk_count = len(chunks)

        # Persist
        self._store.save_document_metadata(metadata)
        self._store.save_chunks(metadata.document_id, chunks)

        manifest = generate_chunk_manifest(chunks)
        self._store.save_artifact(metadata.document_id, "chunk_manifest", manifest)

        logger.info(
            "Document ready: %d pages, %d chunks",
            metadata.page_count, len(chunks),
        )
        return metadata, chunks

    async def answer_questions(
        self,
        document_id: str,
        questions: list[QuestionRequest],
        *,
        mode: Optional[PipelineMode] = None,
        progress_callback: Optional[callable] = None,
    ) -> PipelineRun:
        """Answer questions about a previously parsed document.

        Args:
            document_id: Document ID from process_document.
            questions: Questions to answer.
            mode: Pipeline mode (default from settings).
            progress_callback: Optional progress callback.

        Returns:
            Complete PipelineRun with answers and diagnostics.
        """
        mode = mode or self._settings.pipeline_mode
        start_time = time.perf_counter()

        run = PipelineRun(
            document_id=document_id,
            pipeline_mode=mode.value,
            started_at=datetime.now(timezone.utc),
        )
        submission_writer = IncrementalSubmissionWriter(
            self._settings.runs_dir / f"{run.run_id}_submission.json",
            team=self._settings.team_name,
            notes=self._settings.submission_notes,
        )
        run.submission_path = str(submission_writer.initialize(questions))

        # Load chunks
        chunks = self._store.load_chunks(document_id)
        if not chunks:
            run.warnings.append("No chunks found for document")
            run.completed_at = datetime.now(timezone.utc)
            return run

        metadata = self._store.load_document_metadata(document_id)
        quality = self._store.load_quality_records(document_id)

        if mode == PipelineMode.FULLSCAN_OPERATOR:
            await self._run_fullscan(
                run,
                chunks,
                questions,
                metadata,
                quality,
                progress_callback,
                submission_writer,
            )
        elif mode == PipelineMode.DIRECT_CONTEXT:
            await self._run_direct_context(run, chunks, questions)
        elif mode == PipelineMode.SUMMARY_MAP_REDUCE:
            await self._run_summary_mapreduce(run, chunks, questions)
        elif mode == PipelineMode.REFINE:
            await self._run_refine(run, chunks, questions)

        run.completed_at = datetime.now(timezone.utc)
        run.usage = self._usage_tracker.records
        run.total_input_tokens = self._usage_tracker.total_input_tokens
        run.total_output_tokens = self._usage_tracker.total_output_tokens
        run.total_latency_ms = (time.perf_counter() - start_time) * 1000
        run.submission_path = str(submission_writer.finalize(run))

        self._store.save_run(run)
        return run

    async def _run_fullscan(
        self,
        run: PipelineRun,
        chunks: list[DocumentChunk],
        questions: list[QuestionRequest],
        metadata: Optional[DocumentMetadata],
        quality: list,
        progress_callback: Optional[callable],
        submission_writer: IncrementalSubmissionWriter,
    ) -> None:
        """Run the FULLSCAN_OPERATOR pipeline."""
        # 1. Plan all questions
        logger.info("Planning %d questions", len(questions))
        plans = await self._planner.plan_batch(questions)

        # 2. Exhaustive mapping over ALL chunks
        logger.info("Mapping %d chunks × %d questions", len(chunks), len(plans))
        map_results = await self._batch_mapper.map_all_chunks(
            chunks, plans, progress_callback=progress_callback
        )

        # 3. Process each question
        for plan in plans:
            try:
                answer = await self._process_single_question(
                    plan, chunks, map_results, quality
                )
                run.answers.append(answer)
                submission_writer.update(answer)
            except Exception as exc:
                logger.error("Failed to process question %s: %s", plan.question_id, exc)
                error_answer = PipelineAnswer(
                    question_id=plan.question_id,
                    final_answer=f"Error: {exc}",
                    warnings=[str(exc)],
                    operator=plan.operator,
                    plan=plan,
                )
                run.answers.append(error_answer)
                submission_writer.update(error_answer)

    async def _process_single_question(
        self,
        plan,
        chunks,
        map_results,
        quality,
    ) -> PipelineAnswer:
        """Process a single question through reduction → answer → verify."""
        qid = plan.question_id

        # Filter results for this question
        q_results = [r for r in map_results if r.question_id == qid]

        # Collect evidence items
        all_evidence = []
        for r in q_results:
            all_evidence.extend(r.evidence_items)

        # Deduplicate entities
        entities, dedup_decisions = deduplicate_entities(all_evidence)

        # Build evidence ledger
        ledger = build_ledger(qid, plan, q_results, entities, len(chunks))

        # Run deterministic reducer
        operation_result = reduce(plan, ledger)

        # Coverage verification
        coverage = verify_coverage(plan, chunks, map_results, entities, quality)

        # Generate answer
        final_answer, answer_with_evidence = await self._answer_gen.generate(
            plan, operation_result, all_evidence, coverage
        )

        # Verify claims
        verifications = []
        try:
            final_answer, verifications = await self._verifier.verify_and_repair(
                final_answer, all_evidence, operation_result, qid
            )
        except Exception as exc:
            logger.warning("Verification failed for %s: %s", qid, exc)

        return PipelineAnswer(
            question_id=qid,
            final_answer=final_answer,
            answer_with_evidence=answer_with_evidence,
            operator=plan.operator,
            plan=plan,
            operation_result=operation_result,
            coverage=coverage,
            verification=verifications,
            warnings=operation_result.warnings + coverage.warnings,
        )

    # ----- Baseline modes -----

    async def _run_direct_context(
        self, run, chunks, questions
    ) -> None:
        """Baseline A: stuff document context into prompt and answer in parallel."""
        import asyncio

        full_text = "\n\n".join(c.text for c in chunks)
        # Truncate context to ~30k chars (~7.5k tokens) for fast local inference
        max_chars = 30_000
        context = full_text[:max_chars]

        async def _answer_one(q):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the provided document. "
                        "Be concise and precise."
                    ),
                },
                {"role": "user", "content": f"DOCUMENT:\n{context}\n\nQUESTION: {q.question}"},
            ]
            resp = await self._router.chat(
                messages, stage="answer", max_tokens=1024, question_ids=[q.question_id]
            )
            if resp.usage:
                self._usage_tracker.record(resp.usage)
            return PipelineAnswer(
                question_id=q.question_id,
                final_answer=resp.content,
                answer_with_evidence=resp.content,
            )

        tasks = [_answer_one(q) for q in questions]
        answers = await asyncio.gather(*tasks)
        run.answers.extend(answers)

    async def _run_summary_mapreduce(
        self, run, chunks, questions
    ) -> None:
        """Baseline B: summarize each chunk, then synthesize."""
        import asyncio

        for q in questions:
            summaries: list[str] = []
            sem = asyncio.Semaphore(self._settings.max_concurrent_requests)

            async def summarize(chunk):
                async with sem:
                    msgs = [
                        {
                            "role": "system",
                            "content": (
                                "Summarize this text focusing on answering the "
                                "question. Be concise."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"QUESTION: {q.question}\n\nTEXT:\n{chunk.text}",
                        },
                    ]
                    resp = await self._router.chat(msgs, stage="mapper", max_tokens=512)
                    if resp.usage:
                        self._usage_tracker.record(resp.usage)
                    return resp.content

            tasks = [summarize(c) for c in chunks]
            summaries = await asyncio.gather(*tasks)

            combined = "\n\n".join(f"[Chunk {i+1}]: {s}" for i, s in enumerate(summaries))
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Synthesize these summaries to answer the question "
                        "precisely."
                    ),
                },
                {"role": "user", "content": f"QUESTION: {q.question}\n\nSUMMARIES:\n{combined}"},
            ]
            resp = await self._router.chat(
                msgs,
                stage="answer",
                max_tokens=2048,
                question_ids=[q.question_id],
            )
            if resp.usage:
                self._usage_tracker.record(resp.usage)

            run.answers.append(PipelineAnswer(
                question_id=q.question_id,
                final_answer=resp.content,
                answer_with_evidence=resp.content,
            ))

    async def _run_refine(
        self, run, chunks, questions
    ) -> None:
        """Baseline C: sequential draft refinement."""
        for q in questions:
            draft = ""
            for i, chunk in enumerate(chunks):
                msgs = [
                    {
                        "role": "system",
                        "content": (
                            "Refine the draft answer using new evidence. "
                            "Preserve existing correct info."
                        ),
                    },
                    {"role": "user", "content": (
                        f"QUESTION: {q.question}\n\n"
                        f"CURRENT DRAFT: {draft or '(empty)'}\n\n"
                        f"NEW EVIDENCE (chunk {i+1}/{len(chunks)}):\n{chunk.text}"
                    )},
                ]
                resp = await self._router.chat(msgs, stage="answer", max_tokens=1024)
                if resp.usage:
                    self._usage_tracker.record(resp.usage)
                draft = resp.content

            run.answers.append(PipelineAnswer(
                question_id=q.question_id,
                final_answer=draft,
                answer_with_evidence=draft,
            ))

    async def close(self) -> None:
        """Clean up resources."""
        await self._router.close()
