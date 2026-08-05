"""Regression tests for question-independent V3 document compilation."""

from __future__ import annotations

from app.schemas import DocumentPage
from app.v3.compiler import all_mapping_records, compile_document, parse_number_facts


def test_number_card_binds_each_value_to_following_label():
    text = """# Park in numbers
1721
Highest point: Sandfloegga (m)
7000
Wild reindeer living on the plateau
3430
Area covered (sq km)
"""
    facts = parse_number_facts("record-001", 10, text)

    highest = [fact for fact in facts if fact.field == "highest_point"]
    assert len(highest) == 1
    assert highest[0].value == 1721
    assert highest[0].subject == "Sandfloegga"
    assert not any(fact.value == 7000 and fact.field == "highest_point" for fact in facts)


def test_number_card_normalizes_millions_and_bc_years():
    text = """Park in numbers
15
Annual visitors (million)
475
First recorded eruption (BC)
"""
    facts = parse_number_facts("record-001", 4, text)
    by_field = {fact.field: fact for fact in facts}

    assert by_field["annual_visitors"].value == 15_000_000
    assert by_field["first_recorded_eruption"].value == -475


def test_generic_compiler_preserves_every_nonconsecutive_page():
    pages = [
        DocumentPage(page_number=1, text="A" * 8000),
        DocumentPage(page_number=2, text="B" * 8000),
        DocumentPage(page_number=4, text="C" * 100),
    ]
    document = compile_document("doc", pages)
    records = all_mapping_records(document)

    assert document.record_kind == "segments"
    assert [(record.page_start, record.page_end) for record in records] == [
        (1, 1),
        (2, 2),
        (4, 4),
    ]
    assert "A" * 100 in records[0].text
    assert "B" * 100 in records[1].text
    assert "C" * 100 in records[2].text


def test_repeated_compiler_separates_detected_back_matter():
    pages = []
    for page_number, title in enumerate(
        ("Alpha National Park", "Beta National Park", "Gamma National Park"),
        start=1,
    ):
        pages.append(DocumentPage(
            page_number=page_number,
            text=f"# {title}\nPark in numbers\n100\nArea covered (sq km)",
        ))
    pages.append(DocumentPage(page_number=4, text="# Index\nAlpha 1\nBeta 2"))

    document = compile_document("doc", pages)

    assert document.records[-1].page_end == 3
    assert len(document.supplementary_records) == 1
    assert document.supplementary_records[0].page_start == 4
