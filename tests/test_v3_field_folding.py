"""Folding one measurement's wording variants into a single field."""

from __future__ import annotations

from app.v3.compiler import field_id, merge_synonym_fields
from app.v3.models import CompiledRecord, NumberFact


def _record(ordinal: int, facts: list[tuple[str, float]]) -> CompiledRecord:
    record_id = f"record-{ordinal:03d}"
    return CompiledRecord(
        record_id=record_id,
        ordinal=ordinal,
        title=f"Entity {ordinal}",
        page_start=ordinal,
        page_end=ordinal,
        anchor_page=ordinal,
        text=f"[Page {ordinal}]",
        number_facts=[
            NumberFact(
                record_id=record_id,
                field=field_id(label),
                label=label,
                value=value,
                raw_value=str(value),
                page=ordinal,
                quote=f"{label}: {value}",
            )
            for label, value in facts
        ],
    )


def test_wording_variants_of_one_metric_are_folded():
    # The document names the same measurement two ways, and no record uses
    # both, so the split is a wording difference rather than two metrics.
    records = [_record(index, [("Annual throughput", 100 + index)]) for index in (1, 2, 3)]
    records += [_record(index, [("Throughput", 100 + index)]) for index in (4, 5)]

    mapping = merge_synonym_fields(records)

    assert mapping == {"throughput": "annual_throughput"}
    fields = {fact.field for record in records for fact in record.number_facts}
    assert fields == {"annual_throughput"}


def test_metrics_reported_together_are_never_folded():
    # Both appear in the same record, so they measure different things even
    # though one name contains the other.
    records = [
        _record(index, [("Total area", 10 + index), ("Area", 5 + index)])
        for index in (1, 2)
    ]

    assert merge_synonym_fields(records) == {}


def test_generic_shared_wording_does_not_fold_distinct_metrics():
    # "Number of ..." is a shape the document repeats, not a shared metric:
    # folding on it would merge lakes into villages.
    records = [
        _record(1, [("Number of lakes", 12)]),
        _record(2, [("Number of villages", 4)]),
        _record(3, [("Number of caves", 7)]),
        _record(4, [("Number", 3)]),
    ]

    mapping = merge_synonym_fields(records)

    assert "number_of_lakes" not in mapping
    assert "number_of_villages" not in mapping
    assert mapping.get("number") is None


def test_two_ways_of_writing_one_metric_are_one_field():
    """"Approximate number of visitors annually" and "million visitors per
    year" measure the same thing; everything that differs says how many or
    how often rather than what."""
    from app.v3.compiler import merge_synonym_fields

    records = [
        _record(1, [("Approximate number of visitors annually", 800000)]),
        _record(2, [("Million visitors per year", 15000000)]),
    ]

    mapping = merge_synonym_fields(records)

    assert len(mapping) == 1
    assert len({fact.field for record in records for fact in record.number_facts}) == 1


def test_a_measurement_label_is_not_a_record_title():
    """A chapter opening on its fact card must not be named after the card."""
    from app.v3.compiler import _drop_label_titles

    records = [
        _record(1, [("Area covered (sq km)", 100)]),
        _record(2, [("Area covered (sq km)", 200)]),
    ]
    records[0].title = "Real Park"
    records[1].title = "Area covered (sq km)"

    _drop_label_titles(records)

    assert records[0].title == "Real Park"
    # Left numbered, so profiling can find the name the page actually carries.
    assert records[1].title == "Record 2"
