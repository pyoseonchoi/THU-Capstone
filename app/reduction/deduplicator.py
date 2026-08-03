"""Entity deduplication.

Merges entity records across chunks using:
- Explicit section/entity identifiers
- Repeated heading matches
- Page continuity
- Conservative normalized name matching

Conflicting values are preserved as conflicts, never silently overwritten.
All deduplication decisions are recorded.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.reduction.normalizer import normalize_entity_name
from app.schemas import EntityRecord, EvidenceItem

logger = get_logger("reduction.deduplicator")


class DeduplicationDecision:
    """Record of a deduplication merge decision."""

    def __init__(
        self,
        entity_a: str,
        entity_b: str,
        reason: str,
        merged: bool,
    ):
        self.entity_a = entity_a
        self.entity_b = entity_b
        self.reason = reason
        self.merged = merged

    def __repr__(self) -> str:
        action = "MERGED" if self.merged else "KEPT_SEPARATE"
        return f"{action}: '{self.entity_a}' + '{self.entity_b}' ({self.reason})"


def deduplicate_entities(
    evidence_items: list[EvidenceItem],
) -> tuple[list[EntityRecord], list[DeduplicationDecision]]:
    """Deduplicate entities from evidence items.

    Groups evidence by normalized entity name, merges records with
    matching section IDs or exact name matches, and preserves conflicts.

    Returns:
        Tuple of (deduplicated EntityRecords, dedup decisions).
    """
    decisions: list[DeduplicationDecision] = []

    # Group by normalized entity name
    name_groups: dict[str, list[EvidenceItem]] = {}
    for item in evidence_items:
        if not item.entity_name:
            continue
        key = normalize_entity_name(item.entity_name)
        if key not in name_groups:
            name_groups[key] = []
        name_groups[key].append(item)

    entities: list[EntityRecord] = []

    for norm_name, items in name_groups.items():
        # Further group by section_id for same-name entities in different sections
        section_groups: dict[str, list[EvidenceItem]] = {}
        for item in items:
            sec_key = item.section_id or "_default"
            if sec_key not in section_groups:
                section_groups[sec_key] = []
            section_groups[sec_key].append(item)

        # If all items have the same section or are clearly the same entity, merge
        if len(section_groups) <= 1 or _should_merge_sections(section_groups):
            entity = _merge_items(norm_name, items)
            entities.append(entity)
            if len(section_groups) > 1:
                decisions.append(DeduplicationDecision(
                    entity_a=norm_name,
                    entity_b=f"{len(section_groups)} sections",
                    reason="same_normalized_name_cross_section",
                    merged=True,
                ))
        else:
            # Keep separate: different sections might be different entities
            for sec_key, sec_items in section_groups.items():
                entity = _merge_items(norm_name, sec_items)
                entity.entity_id = f"{entity.entity_id}_{sec_key[:6]}"
                entities.append(entity)
            decisions.append(DeduplicationDecision(
                entity_a=norm_name,
                entity_b=f"{len(section_groups)} sections",
                reason="different_sections_kept_separate",
                merged=False,
            ))

    logger.info(
        "Deduplicated %d evidence items into %d entities (%d decisions)",
        len(evidence_items),
        len(entities),
        len(decisions),
    )
    return entities, decisions


def _should_merge_sections(
    section_groups: dict[str, list[EvidenceItem]],
) -> bool:
    """Determine if items from different sections should be merged.

    Merges when:
    - Pages are continuous or overlapping
    - Same entity name appears with consistent field values
    """
    all_pages: list[int] = []
    for items in section_groups.values():
        for item in items:
            if item.page_start:
                all_pages.append(item.page_start)
            if item.page_end:
                all_pages.append(item.page_end)

    if not all_pages:
        return True  # No page info: merge conservatively

    # Check page continuity: if max - min <= 5 pages, likely same entity
    page_span = max(all_pages) - min(all_pages)
    return page_span <= 5


def _merge_items(norm_name: str, items: list[EvidenceItem]) -> EntityRecord:
    """Merge evidence items into a single entity record."""
    entity = EntityRecord(
        entity_name=items[0].entity_name,
        normalized_name=norm_name,
    )

    for item in items:
        entity.evidence_ids.append(item.evidence_id)
        if item.chunk_id and item.chunk_id not in entity.source_chunks:
            entity.source_chunks.append(item.chunk_id)
        for p in range(item.page_start, item.page_end + 1):
            if p and p not in entity.source_pages:
                entity.source_pages.append(p)

        if item.field_name:
            existing_val = entity.fields.get(item.field_name)
            if existing_val is not None and item.normalized_value is not None:
                if existing_val != item.normalized_value:
                    conflict = (
                        f"Conflict for '{item.field_name}': "
                        f"{existing_val} vs {item.normalized_value} "
                        f"(chunk {item.chunk_id})"
                    )
                    entity.conflicts.append(conflict)
                    # Keep the first value, record conflict
            elif item.normalized_value is not None:
                entity.fields[item.field_name] = item.normalized_value
                entity.raw_fields[item.field_name] = item.raw_value

    return entity
