"""Answering from the compiled registry rather than re-reading the prose."""

from __future__ import annotations

from app.schemas import QuestionRequest
from app.v3.models import CompiledDocument, CompiledRecord, NumberFact
from app.v3.question_compiler import compile_question
from app.v3.structured_executor import execute_structured


def _record(
    ordinal: int,
    title: str,
    country: str,
    facts: list[tuple[str, str, float, str]] = (),
    text: str = "",
) -> CompiledRecord:
    record_id = f"record-{ordinal:03d}"
    return CompiledRecord(
        record_id=record_id,
        ordinal=ordinal,
        title=title,
        country=country,
        page_start=ordinal,
        page_end=ordinal,
        anchor_page=ordinal,
        text=text or f"[Page {ordinal}]\n{title}",
        number_facts=[
            NumberFact(
                record_id=record_id,
                field=field,
                label=label,
                value=value,
                raw_value=str(value),
                unit=unit,
                page=ordinal,
                quote=f"{label}: {value}",
            )
            for field, label, value, unit in facts
        ],
    )


def _plan(question: str, category: str, document: CompiledDocument):
    return compile_question(
        QuestionRequest(question_id="q", question=question, category=category),
        document,
        "",
    )


def _cycle_document(records: list[CompiledRecord]) -> CompiledDocument:
    """A registry that failed its integrity check but counted itself."""
    return CompiledDocument(
        document_id="d",
        record_kind="repeated_entity",
        entity_label="park",
        records=records,
        registry_trusted=False,
        registry_signals={
            "fact_sections": len(records) - 2,
            "segmentation": "boilerplate_cycle",
            "cycle_markers": len(records),
        },
    )


def test_group_counts_come_from_the_registry_when_it_counted_itself():
    # The integrity check failed on fact cards, but a marker repeated once per
    # chapter still fixes how many chapters there are.
    document = _cycle_document([
        _record(1, "Alpha Park", "Croatia"),
        _record(2, "Beta Park", "Croatia"),
        _record(3, "Gamma Park", "Montenegro"),
        _record(4, "Delta Park", "Norway"),
    ])

    result = execute_structured(
        _plan("How many of the parks are in Croatia and how many are in Montenegro?",
              "aggregation", document),
        document,
    )

    assert result is not None
    assert "Croatia has 2" in result.answer
    assert "Montenegro has 1" in result.answer


def test_a_registry_that_never_counted_itself_stays_silent():
    document = _cycle_document([_record(1, "Alpha Park", "Croatia")])
    document.registry_signals = {"segmentation": "fact_sections"}

    assert execute_structured(
        _plan("How many of the parks are in Croatia and how many are in Montenegro?",
              "aggregation", document),
        document,
    ) is None


def test_dated_needle_reads_the_bound_fact_not_the_first_number():
    # A fact card has no sentence punctuation, so scanning its text finds the
    # area before the year the question asks for.
    document = _cycle_document([
        _record(
            1,
            "Alpha Park",
            "Italy",
            facts=[
                ("area_covered", "Area covered (sq km)", 581.0, "sq km"),
                ("first_recorded_eruption", "(BC) First recorded eruption", -475.0, "year_bc"),
            ],
            text="[Page 1]\nAlpha Park 581 Area covered (sq km) "
                 "475 (BC) First recorded eruption",
        ),
    ])

    result = execute_structured(
        _plan("In what year is Alpha Park's first recorded eruption dated?",
              "needle", document),
        document,
    )

    assert result is not None
    assert "475 BC" in result.answer
    assert "581" not in result.answer


def test_unit_outlier_ignores_a_record_holding_two_units_for_one_field():
    # Both figures sit in one record, so its boundary swallowed a neighbour's
    # card and the odd unit belongs to a record we cannot name.
    document = _cycle_document([
        _record(1, "Alpha Park", "Norway", facts=[
            ("area_covered", "Area covered (sq km)", 1310.0, "sq km"),
            ("area_covered", "Area covered (sq miles)", 1151.0, "sq miles"),
        ]),
        _record(2, "Beta Park", "Norway", facts=[
            ("area_covered", "Area covered (sq km)", 900.0, "sq km"),
        ]),
        _record(3, "Gamma Park", "Norway", facts=[
            ("area_covered", "Area covered (sq km)", 700.0, "sq km"),
        ]),
    ])

    assert execute_structured(
        _plan("Every park reports its area, but one park is reported in "
              "different units from all the others. Which park, and what units?",
              "contradiction", document),
        document,
    ) is None
