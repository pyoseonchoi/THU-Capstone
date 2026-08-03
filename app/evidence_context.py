"""Shared formatting for bounded, provenance-aware evidence context."""

from __future__ import annotations

from collections.abc import Callable

from app.schemas import EvidenceItem


def format_evidence_context(
    evidence_items: list[EvidenceItem],
    formatter: Callable[[EvidenceItem], str],
    *,
    preferred_ids: list[str] | None = None,
    max_characters: int = 24_000,
) -> str:
    """Format evidence without an arbitrary item-count cutoff.

    Evidence used by the deterministic operation is placed first. If the
    model context budget is reached, the omission is explicitly recorded.
    """
    preferred = set(preferred_ids or [])
    ordered = sorted(
        evidence_items,
        key=lambda item: (item.evidence_id not in preferred, item.page_start),
    )
    lines: list[str] = []
    used_characters = 0
    included_ids: set[str] = set()

    for item in ordered:
        if item.evidence_id in included_ids:
            continue
        line = formatter(item)
        additional = len(line) + 1
        if lines and used_characters + additional > max_characters:
            continue
        lines.append(line)
        used_characters += additional
        included_ids.add(item.evidence_id)

    omitted = len({item.evidence_id for item in evidence_items} - included_ids)
    if omitted:
        lines.append(
            f"[CONTEXT BUDGET: {omitted} additional evidence items omitted; "
            "the complete deterministic operation result remains authoritative.]"
        )
    return "\n".join(lines)
