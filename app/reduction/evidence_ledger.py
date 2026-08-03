"""Evidence ledger: collects and indexes evidence for a question."""

from __future__ import annotations

from app.logging_config import get_logger
from app.schemas import (
    ChunkMapResult,
    EntityRecord,
    EvidenceLedger,
    QueryPlan,
)

logger = get_logger("reduction.evidence_ledger")


def build_ledger(
    question_id: str,
    plan: QueryPlan,
    map_results: list[ChunkMapResult],
    entities: list[EntityRecord],
    total_chunks: int,
) -> EvidenceLedger:
    """Build an evidence ledger from map results.

    Collects all evidence items, builds the topic coverage matrix
    for ABSENCE queries, and records per-chunk statuses.

    Args:
        question_id: The question this ledger is for.
        plan: The query plan.
        map_results: All chunk map results for this question.
        entities: Deduplicated entity records.
        total_chunks: Total number of document chunks.

    Returns:
        A populated EvidenceLedger.
    """
    ledger = EvidenceLedger(
        question_id=question_id,
        plan=plan,
        total_chunks=total_chunks,
        entities=entities,
    )

    # Collect evidence items and chunk statuses
    for result in map_results:
        if result.question_id != question_id:
            continue
        ledger.chunk_statuses[result.chunk_id] = result.extraction_status
        ledger.items.extend(result.evidence_items)
        ledger.conflicts.extend(result.conflicts)

        # Build topic matrix for ABSENCE queries
        for ta in result.topic_assessments:
            if ta.topic not in ledger.topic_matrix:
                ledger.topic_matrix[ta.topic] = {}
            ledger.topic_matrix[ta.topic][result.chunk_id] = ta.level.value

    return ledger
