"""Run-scoped canonical fact store built from exhaustive mapper output."""

from __future__ import annotations

from dataclasses import dataclass

from app.field_names import canonical_field_name
from app.schemas import ChunkMapResult, EvidenceItem, Operator, QueryPlan


@dataclass
class FactStore:
    """Share typed facts across questions after every chunk was mapped."""

    items: list[EvidenceItem]

    @classmethod
    def from_results(cls, results: list[ChunkMapResult]) -> "FactStore":
        unique: dict[tuple, EvidenceItem] = {}
        for result in results:
            for item in result.evidence_items:
                key = (
                    item.entity_name.casefold().strip(),
                    canonical_field_name(item.field_name),
                    item.raw_value.casefold().strip(),
                    item.page_start,
                    item.page_end,
                )
                unique.setdefault(key, item)
        return cls(items=list(unique.values()))

    def for_plan(
        self,
        plan: QueryPlan,
        own_items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        """Enrich structured operations with matching facts from other questions."""
        if plan.operator in {
            Operator.ABSENCE,
            Operator.MULTI_HOP,
            Operator.GENERAL_SYNTHESIS,
        }:
            return own_items

        allowed_fields = {
            *plan.target_fields,
            *(field.field_name for field in plan.extraction_fields),
            *(condition.field for condition in plan.conditions),
        }
        allowed_fields.discard("")
        selected = list(own_items)
        seen = {item.evidence_id for item in selected}
        for item in self.items:
            if item.evidence_id in seen:
                continue
            if canonical_field_name(item.field_name) in allowed_fields:
                selected.append(item)
                seen.add(item.evidence_id)
        return selected
