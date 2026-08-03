"""Batch mapper: exhaustive async mapping over ALL document chunks.

Uses asyncio with a configurable semaphore for concurrency control.
Never skips chunks. Never stops early. Every chunk gets a terminal status.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.config import Settings
from app.logging_config import get_logger
from app.mapping.evidence_mapper import EvidenceMapper
from app.schemas import (
    ChunkMapResult,
    DocumentChunk,
    ExtractionStatus,
    QueryPlan,
)

logger = get_logger("mapping.batch_mapper")


class BatchMapper:
    """Exhaustively maps every chunk for a batch of questions."""

    def __init__(
        self,
        mapper: EvidenceMapper,
        settings: Settings,
    ):
        self._mapper = mapper
        self._max_concurrent = settings.max_concurrent_requests
        self._max_retries = settings.max_retries
        self._batch_size = settings.question_batch_size

    async def map_all_chunks(
        self,
        chunks: list[DocumentChunk],
        plans: list[QueryPlan],
        *,
        progress_callback: Optional[callable] = None,
    ) -> list[ChunkMapResult]:
        """Map every chunk against all questions in batches.

        Args:
            chunks: All document chunks (exhaustive — never skip any).
            plans: Query plans for the current question batch.
            progress_callback: Optional callback(processed, total, failed).

        Returns:
            Complete list of ChunkMapResults.
        """
        semaphore = asyncio.Semaphore(self._max_concurrent)
        all_results: list[ChunkMapResult] = []
        failed_chunks: list[tuple[DocumentChunk, list[QueryPlan]]] = []
        processed = 0
        total = len(chunks)

        async def _process_chunk(
            chunk: DocumentChunk, batch: list[QueryPlan]
        ) -> list[ChunkMapResult]:
            async with semaphore:
                try:
                    results = await self._mapper.map_chunk(chunk, batch)
                except Exception as exc:
                    logger.exception(
                        "Unexpected mapper task failure for chunk %s",
                        chunk.chunk_id,
                    )
                    return self._failure_results(chunk, batch, exc)
                return self._complete_results(chunk, batch, results)

        # Split questions into batches
        plan_batches = self._split_plans(plans)
        total_work = total * len(plan_batches)
        if progress_callback:
            progress_callback(0, total_work, 0)

        for plan_batch in plan_batches:
            # Create tasks for all chunks with this question batch
            tasks = [
                asyncio.create_task(_process_chunk(chunk, plan_batch))
                for chunk in chunks
            ]

            for task in asyncio.as_completed(tasks):
                try:
                    results = await task
                    all_results.extend(results)

                    # Track failures for retry
                    for r in results:
                        if r.extraction_status in (
                            ExtractionStatus.LLM_FAILED,
                            ExtractionStatus.PARSE_FAILED,
                        ):
                            # Find the chunk for retry
                            chunk = next(
                                (c for c in chunks if c.chunk_id == r.chunk_id),
                                None,
                            )
                            if chunk:
                                plan = next(
                                    (p for p in plan_batch if p.question_id == r.question_id),
                                    None,
                                )
                                if plan:
                                    failed_chunks.append((chunk, [plan]))
                except Exception as exc:
                    # _process_chunk is defensive, so this is a last-resort guard.
                    logger.exception("Unexpected batch task error: %s", exc)

                processed += 1
                if progress_callback:
                    fail_count = len(failed_chunks)
                    progress_callback(processed, total_work, fail_count)

        # Retry failed chunks
        if failed_chunks:
            logger.info("Retrying %d failed chunk-question pairs", len(failed_chunks))
            retry_results = await self._retry_failed(failed_chunks)
            # Replace failed results with retry results
            all_results = self._merge_retry_results(all_results, retry_results)

        all_results = self._ensure_complete_matrix(chunks, plans, all_results)

        logger.info(
            "Exhaustive mapping complete: %d total results from %d chunks × %d questions",
            len(all_results),
            len(chunks),
            len(plans),
        )
        return all_results

    def _split_plans(self, plans: list[QueryPlan]) -> list[list[QueryPlan]]:
        """Split plans into batches of configurable size."""
        batches: list[list[QueryPlan]] = []
        for i in range(0, len(plans), self._batch_size):
            batches.append(plans[i : i + self._batch_size])
        return batches

    async def _retry_failed(
        self,
        failed: list[tuple[DocumentChunk, list[QueryPlan]]],
    ) -> list[ChunkMapResult]:
        """Retry failed chunk mappings."""
        retry_results: list[ChunkMapResult] = []
        semaphore = asyncio.Semaphore(self._max_concurrent)

        for attempt in range(1, self._max_retries + 1):
            if not failed:
                break
            logger.info("Retry attempt %d for %d failures", attempt, len(failed))
            still_failed: list[tuple[DocumentChunk, list[QueryPlan]]] = []

            async def _retry(chunk, plans):
                async with semaphore:
                    try:
                        results = await self._mapper.map_chunk(chunk, plans)
                    except Exception as exc:
                        logger.exception(
                            "Retry task failed for chunk %s", chunk.chunk_id
                        )
                        results = self._failure_results(chunk, plans, exc)
                    return chunk, self._complete_results(chunk, plans, results)

            tasks = [
                asyncio.create_task(_retry(chunk, plans))
                for chunk, plans in failed
            ]

            for task in asyncio.as_completed(tasks):
                try:
                    orig_chunk, results = await task
                    # Find original plans for this chunk
                    orig_chunk_plans = next(
                        (ps for c, ps in failed if c.chunk_id == orig_chunk.chunk_id),
                        [],
                    )
                    for r in results:
                        if r.extraction_status in (
                            ExtractionStatus.LLM_FAILED,
                            ExtractionStatus.PARSE_FAILED,
                        ):
                            plan = next(
                                (p for p in orig_chunk_plans if p.question_id == r.question_id),
                                None,
                            )
                            if plan:
                                still_failed.append((orig_chunk, [plan]))
                        else:
                            retry_results.append(r)
                except Exception as exc:
                    logger.exception("Unexpected retry task error: %s", exc)

            failed = still_failed

        return retry_results

    def _merge_retry_results(
        self,
        original: list[ChunkMapResult],
        retries: list[ChunkMapResult],
    ) -> list[ChunkMapResult]:
        """Replace failed results with successful retries."""
        retry_map: dict[tuple[str, str], ChunkMapResult] = {}
        for r in retries:
            key = (r.chunk_id, r.question_id)
            if r.extraction_status not in (
                ExtractionStatus.LLM_FAILED,
                ExtractionStatus.PARSE_FAILED,
            ):
                retry_map[key] = r

        merged: list[ChunkMapResult] = []
        for r in original:
            key = (r.chunk_id, r.question_id)
            if key in retry_map:
                merged.append(retry_map.pop(key))
            else:
                merged.append(r)

        # Add any remaining retry results not in original
        merged.extend(retry_map.values())
        return merged

    @staticmethod
    def _failure_results(
        chunk: DocumentChunk,
        plans: list[QueryPlan],
        exc: Exception,
    ) -> list[ChunkMapResult]:
        return [
            ChunkMapResult(
                chunk_id=chunk.chunk_id,
                question_id=plan.question_id,
                extraction_status=ExtractionStatus.LLM_FAILED,
                uncertainty_notes=str(exc),
            )
            for plan in plans
        ]

    @classmethod
    def _complete_results(
        cls,
        chunk: DocumentChunk,
        plans: list[QueryPlan],
        results: list[ChunkMapResult],
    ) -> list[ChunkMapResult]:
        """Return exactly one terminal result for every requested question."""
        by_question = {
            result.question_id: result
            for result in results
            if result.question_id in {plan.question_id for plan in plans}
        }
        completed: list[ChunkMapResult] = []
        for plan in plans:
            result = by_question.get(plan.question_id)
            if result is None:
                result = ChunkMapResult(
                    chunk_id=chunk.chunk_id,
                    question_id=plan.question_id,
                    extraction_status=ExtractionStatus.PARSE_FAILED,
                    uncertainty_notes="Mapper returned no terminal result",
                )
            else:
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
            completed.append(result)
        return completed

    @classmethod
    def _ensure_complete_matrix(
        cls,
        chunks: list[DocumentChunk],
        plans: list[QueryPlan],
        results: list[ChunkMapResult],
    ) -> list[ChunkMapResult]:
        """Deduplicate results and fill any missing chunk-question pair."""
        result_map = {
            (result.chunk_id, result.question_id): result
            for result in results
        }
        complete: list[ChunkMapResult] = []
        for chunk in chunks:
            chunk_results = [
                result_map[(chunk.chunk_id, plan.question_id)]
                for plan in plans
                if (chunk.chunk_id, plan.question_id) in result_map
            ]
            complete.extend(cls._complete_results(chunk, plans, chunk_results))
        return complete
