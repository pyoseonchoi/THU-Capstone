"""Coverage verification before final answer generation.

Ensures all chunks are processed and coverage requirements are met.
"""

from __future__ import annotations

import re

from app.logging_config import get_logger
from app.schemas import (
    ChunkMapResult,
    CoverageReport,
    DocumentChunk,
    EntityRecord,
    ExtractionStatus,
    PageQualityRecord,
    QueryPlan,
)

logger = get_logger("verification.coverage_verifier")


def _required_percentage(plan: QueryPlan) -> float:
    required = plan.required_coverage.strip().lower()
    if required == "all":
        return 100.0
    match = re.search(r"\d+(?:\.\d+)?", required)
    if not match:
        return 100.0
    return min(100.0, max(0.0, float(match.group(0))))


def verify_coverage(
    plan: QueryPlan,
    chunks: list[DocumentChunk],
    map_results: list[ChunkMapResult],
    entities: list[EntityRecord],
    page_quality: list[PageQualityRecord],
) -> CoverageReport:
    """Build a coverage report for a question.

    Checks that all chunks have been processed and all required
    entities have the necessary fields.
    """
    total_chunks = len(chunks)
    chunk_ids = {c.chunk_id for c in chunks}

    # Count statuses from map results for this question
    qid = plan.question_id
    q_results = [r for r in map_results if r.question_id == qid]

    result_by_chunk: dict[str, ExtractionStatus] = {}
    for r in q_results:
        result_by_chunk[r.chunk_id] = r.extraction_status

    successful = sum(
        1 for s in result_by_chunk.values()
        if s == ExtractionStatus.EVIDENCE_FOUND
    )
    no_evidence = sum(
        1 for s in result_by_chunk.values()
        if s == ExtractionStatus.NO_EVIDENCE
    )
    uncertain = sum(
        1 for s in result_by_chunk.values()
        if s == ExtractionStatus.UNCERTAIN
    )
    failed = sum(
        1 for s in result_by_chunk.values()
        if s in (ExtractionStatus.LLM_FAILED, ExtractionStatus.PARSE_FAILED)
    )

    # Check for missing chunks (chunks with no result at all)
    mapped_chunks = set(result_by_chunk.keys())
    missing_chunks = chunk_ids - mapped_chunks

    # Page-level metrics
    total_pages = len(page_quality)
    parsed_pages = sum(
        1 for p in page_quality if p.extraction_status.value == "ok"
    )
    unresolved_pages = sum(
        1 for p in page_quality
        if p.extraction_status.value in ("no_text", "unresolved")
    )

    # Entity field completeness
    target_field = plan.target_fields[0] if plan.target_fields else ""
    missing_field_count = 0
    if target_field:
        for entity in entities:
            if target_field not in entity.fields:
                missing_field_count += 1

    # Topic coverage for ABSENCE questions
    topic_coverage: dict[str, str] = {}
    if plan.candidate_topics:
        for topic in plan.candidate_topics:
            topic_coverage[topic] = "unknown"

    # Calculate coverage percentage
    processed = successful + no_evidence
    coverage_pct = (processed / total_chunks * 100) if total_chunks > 0 else 0.0
    required_pct = _required_percentage(plan)
    requirement_met = coverage_pct >= required_pct

    warnings: list[str] = []
    if missing_chunks:
        warnings.append(f"{len(missing_chunks)} chunks have no mapping result")
    if failed > 0:
        warnings.append(f"{failed} chunks failed processing")
    if uncertain > 0:
        warnings.append(f"{uncertain} chunks have uncertain results")
    if unresolved_pages > 0:
        warnings.append(f"{unresolved_pages} pages could not be extracted")
    if missing_field_count > 0:
        warnings.append(
            f"{missing_field_count} entities missing field '{target_field}'"
        )
    if not requirement_met:
        warnings.append(
            f"Coverage requirement not met: {coverage_pct:.2f}% < {required_pct:.2f}%"
        )

    report = CoverageReport(
        total_pages=total_pages,
        parsed_pages=parsed_pages,
        unresolved_pages=unresolved_pages,
        total_chunks=total_chunks,
        successful_mappings=successful,
        no_evidence_chunks=no_evidence,
        uncertain_chunks=uncertain,
        failed_chunks=failed + len(missing_chunks),
        detected_entity_count=len(entities),
        entities_missing_fields=missing_field_count,
        topic_coverage=topic_coverage,
        coverage_percentage=round(coverage_pct, 2),
        required_percentage=required_pct,
        requirement_met=requirement_met,
        mapped_chunks=len(mapped_chunks & chunk_ids),
        missing_chunks=len(missing_chunks),
        warnings=warnings,
    )

    logger.info(
        "Coverage for q=%s: %.1f%% (%d/%d chunks), %d entities, %d warnings",
        qid, coverage_pct, processed, total_chunks,
        len(entities), len(warnings),
    )
    return report
