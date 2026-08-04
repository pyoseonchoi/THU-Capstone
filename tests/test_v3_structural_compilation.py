"""Regression tests for large-document structural compilation paths."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import LLMResponse
from app.llm.usage_tracker import UsageTracker
from app.schemas import DocumentPage, QuestionRequest
from app.v3.compiler import compile_document
from app.v3.models import CompiledDocument, ContentsEntry
from app.v3.question_compiler import compile_question
from app.v3.structure_augmenter import StructureAugmenter
from app.v3.structured_executor import execute_structured


def _inline_project_pages() -> list[DocumentPage]:
    text = " ".join((
        "PROFILE 01: Northlight Project. COUNTRY: Boreal. "
        "PROJECT IN NUMBERS: monitored project area 100 sq km; highest crest or "
        "operating point 3,050 metres; annual output 800 gigawatt-hours; commissioned "
        "1932. Northlight is called Boreal's highest project. The Blue Lantern "
        "Laboratory began as a navigation hut in 1908.",
        "PROFILE 02: Whitecap Project. COUNTRY: Boreal. "
        "PROJECT IN NUMBERS: monitored project area 200 sq km; highest crest or "
        "operating point 3,180 metres; annual output 900 gigawatt-hours; commissioned "
        "1950.",
        "PROFILE 03: Ironwood Canal Project. COUNTRY: Faron. "
        "PROJECT IN NUMBERS: monitored project area 300 sq km; highest crest or "
        "operating point 1,540 metres; annual output 2,700 gigawatt-hours; commissioned "
        "1948. The Orun willow, estimated to be 1,620 years old, grows beside "
        "Ironwood Canal in Faron. The first recorded canal flood is dated to 612 BC.",
    ))
    return [DocumentPage(page_number=1, text=text)]


def _hdi_table_pages() -> list[DocumentPage]:
    heading = (
        "T A B L E 1 Human development components Human Development Index "
        "Life expectancy Gross national income per capita "
    )
    return [
        DocumentPage(
            page_number=10,
            text=(
                heading
                + "Very high human development "
                "1 Alpha 0.950 81.0 15.0 12.0 50,000 "
                "2 Beta 0.940 84.0 14.0 11.0 60,000 "
                "High human development "
                "3 Gamma 0.800 78.0 13.0 10.0 40,000"
            ),
        ),
        DocumentPage(
            page_number=11,
            text=(
                heading
                + "Low human development "
                "4 Delta 0.500 65.0 9.0 6.0 5,000 "
                "Other countries or territories Monaco 0.999 99.0 20.0 20.0 999,999"
            ),
        ),
    ]


def _request(question_id: str, question: str, category: str) -> QuestionRequest:
    return QuestionRequest(
        question_id=question_id,
        question=question,
        category=category,
    )


def test_inline_profiles_compile_complete_schema_and_non_profile_needles():
    document = compile_document("projects", _inline_project_pages())

    assert document.registry_trusted is True
    assert len(document.records) == 3
    assert document.registry_signals["missing_ordinals"] == 0
    assert document.field_catalog == [
        "annual_output",
        "area",
        "establishment_year",
        "highest_point",
    ]

    questions = [
        _request(
            "rank",
            "The guide contradicts Northlight's claim to be Boreal's highest project. "
            "What is the claim and counterevidence?",
            "contradiction",
        ),
        _request(
            "origin",
            "What did the Blue Lantern Laboratory begin as, and in what year?",
            "needle",
        ),
        _request(
            "age",
            "What is the estimated age of the Orun willow, and at which project and "
            "country is it found?",
            "needle",
        ),
        _request(
            "flood",
            "In what year is Ironwood Canal's first recorded canal flood dated?",
            "needle",
        ),
    ]
    answers = [
        execute_structured(compile_question(question, document), document)
        for question in questions
    ]

    assert all(answer is not None for answer in answers)
    assert "3,180" in answers[0].answer
    assert answers[1].answer == (
        "The Blue Lantern Laboratory began as a navigation hut in 1908."
    )
    assert "1,620 years old" in answers[2].answer
    assert "Ironwood Canal Project in Faron" in answers[2].answer
    assert answers[3].answer == "The first recorded canal flood is dated to 612 BC."


def test_ranked_table_grouping_and_argmax_exclude_unranked_entries():
    document = compile_document("report", _hdi_table_pages())
    table = document.tables[0]

    assert table.trusted is True
    assert [row.label for row in table.rows] == ["Alpha", "Beta", "Gamma", "Delta"]
    assert all(row.rank is not None for row in table.rows)

    group_plan = compile_question(_request(
        "groups",
        "In Statistical Annex Table 1, how many ranked entries fall into each Human "
        "Development Index group, and how many are there in total?",
        "aggregation",
    ), document)
    life_plan = compile_question(_request(
        "life",
        "Among the ranked entries in Statistical Annex Table 1, which country has "
        "the highest life expectancy in 2023?",
        "superlative",
    ), document)

    grouped = execute_structured(group_plan, document)
    life = execute_structured(life_plan, document)
    assert grouped is not None
    assert "total 4" in grouped.answer
    assert "Very high human development: 2" in grouped.answer
    assert "High human development: 1" in grouped.answer
    assert "Low human development: 1" in grouped.answer
    assert life is not None
    assert "Beta" in life.answer
    assert "84.0 years" in life.answer
    assert "Monaco" not in life.answer


def test_contents_index_uses_body_captions_to_recover_interleaved_columns():
    pages = [
        DocumentPage(
            page_number=1,
            text=(
                "Contents BOXES 1.1 2.1 FIGURES O.1 1.1 TABLES 1.1 "
                "SPOTLIGHTS 3.1 4.1 Contents ix"
            ),
        ),
        DocumentPage(page_number=2, text="OV E R V I E W Main report"),
        DocumentPage(
            page_number=3,
            text=(
                "Box 1.1 First box. Figure O.1 Overview chart. Spotlight 3.1 Agency. "
                "Table 1.1 Comparison."
            ),
        ),
        DocumentPage(
            page_number=4,
            text=(
                "Box 2.1 Second box. Figure 1.1 Chapter chart. Spotlight 4.1 Care."
            ),
        ),
    ]

    document = compile_document("report", pages)

    assert document.contents_trusted is True
    assert document.contents_pages == [1]
    assert {
        category: [
            entry.identifier
            for entry in document.contents_entries
            if entry.category == category
        ]
        for category in ("boxes", "spotlights", "figures", "tables")
    } == {
        "boxes": ["1.1", "2.1"],
        "spotlights": ["3.1", "4.1"],
        "figures": ["O.1", "1.1"],
        "tables": ["1.1"],
    }


class _FakeRouter:
    def __init__(self, payload: dict[str, list[str]]) -> None:
        self.payload = payload
        self.calls = 0

    def get_model(self, stage: str) -> str:
        assert stage == "planner"
        return "fake-planner"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=json.dumps(self.payload))


@pytest.mark.asyncio
async def test_contents_augmentation_validates_source_and_uses_cache(tmp_path):
    source = (
        "[Page 2]\nCONTENTS BOXES 1.1 S1.2 SPOTLIGHTS S2.1 "
        "FIGURES O.1 1.1 2.1 TABLES 1 2"
    )
    payload = {
        "boxes": ["1.1", "S1.2"],
        "spotlights": ["S2.1"],
        "figures": ["O.1", "1.1", "2.1"],
        "tables": ["1", "2"],
    }
    router = _FakeRouter(payload)
    settings = Settings(data_dir=tmp_path)
    augmenter = StructureAugmenter(router, UsageTracker(), settings)
    first = CompiledDocument(
        document_id="report",
        contents_text=source,
        contents_pages=[2],
    )

    await augmenter.augment(first)

    assert first.contents_trusted is True
    assert router.calls == 1
    assert len(first.contents_entries) == 8

    second = CompiledDocument(
        document_id="report",
        contents_text=source,
        contents_pages=[2],
    )
    await augmenter.augment(second)

    assert second.contents_trusted is True
    assert router.calls == 1


@pytest.mark.asyncio
async def test_contents_augmentation_rejects_identifiers_not_in_source(tmp_path):
    router = _FakeRouter({
        "boxes": ["1.1", "9.9"],
        "spotlights": ["S2.1"],
        "figures": ["O.1"],
        "tables": ["1"],
    })
    document = CompiledDocument(
        document_id="report",
        contents_text=(
            "CONTENTS BOXES 1.1 SPOTLIGHTS S2.1 FIGURES O.1 TABLES 1"
        ),
        contents_entries=[
            ContentsEntry(category="boxes", identifier="1.1"),
            ContentsEntry(category="boxes", identifier="9.9"),
            ContentsEntry(category="spotlights", identifier="S2.1"),
            ContentsEntry(category="figures", identifier="O.1"),
            ContentsEntry(category="tables", identifier="1"),
        ],
    )

    await StructureAugmenter(
        router,
        UsageTracker(),
        Settings(data_dir=tmp_path),
    ).augment(document)

    assert document.contents_trusted is False
    assert "Contents index failed source validation" in document.warnings
