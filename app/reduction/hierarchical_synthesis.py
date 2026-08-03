"""Section-level deterministic reduction for document-wide synthesis."""

from __future__ import annotations

from collections import defaultdict

from app.schemas import EvidenceLedger, OperationResult, QueryPlan


def reduce_hierarchical_synthesis(
    plan: QueryPlan,
    ledger: EvidenceLedger,
    *,
    claims_per_section: int = 8,
) -> OperationResult:
    """Reduce mapped evidence into section summaries without losing provenance."""
    grouped: dict[str, list] = defaultdict(list)
    for item in ledger.items:
        key = item.section_id or f"pages-{item.page_start}-{item.page_end}"
        grouped[key].append(item)

    table: list[dict] = []
    supporting_ids: list[str] = []
    included_entities: list[str] = []
    for section_id, items in sorted(
        grouped.items(),
        key=lambda pair: min(item.page_start for item in pair[1]),
    ):
        ordered = sorted(
            items,
            key=lambda item: (-item.relevance, -item.confidence, item.page_start),
        )
        claims: list[dict] = []
        seen_claims: set[str] = set()
        for item in ordered:
            claim = (item.claim or item.raw_value or item.exact_quote).strip()
            key = claim.casefold()
            if not claim or key in seen_claims:
                continue
            claims.append({
                "entity": item.entity_name,
                "field": item.field_name,
                "claim": claim,
                "page": item.page_start,
                "evidence_id": item.evidence_id,
            })
            seen_claims.add(key)
            supporting_ids.append(item.evidence_id)
            if item.entity_name and item.entity_name not in included_entities:
                included_entities.append(item.entity_name)
            if len(claims) >= claims_per_section:
                break
        if claims:
            table.append({
                "section_id": section_id,
                "page_start": min(item.page_start for item in items),
                "page_end": max(item.page_end for item in items),
                "claims": claims,
                "omitted_claims": max(0, len(items) - len(claims)),
            })

    return OperationResult(
        question_id=plan.question_id,
        operator=plan.operator,
        result_table=table,
        included_entities=included_entities,
        computation_detail=(
            f"Hierarchical synthesis over {len(table)} sections and "
            f"{len(ledger.items)} evidence items"
        ),
        supporting_evidence_ids=list(dict.fromkeys(supporting_ids)),
        source_pages=sorted({
            page
            for row in table
            for page in range(row["page_start"], row["page_end"] + 1)
        }),
    )
