"""Regression tests for partial failures in the V3 exhaustive path."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.schemas import OperationResult, Operator, PipelineAnswer, UsageRecord
from app.submission import answer_to_submission
from app.v3.answerer import V3Answerer
from app.v3.exhaustive_mapper import ExhaustiveMapper, _validated_topic_level
from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    EvidenceCandidate,
    EvidencePacket,
    NumberFact,
    Strategy,
    TopicAssessment,
    V3MapResult,
    V3QuestionPlan,
)
from app.v3.reducers import reduce_absence
from app.v3.structured_executor import execute_structured


def _lookup_plan() -> V3QuestionPlan:
    return V3QuestionPlan(
        question_id="q1",
        question="What did the Icehotel start out as, and in what year?",
        category="needle",
        strategy=Strategy.EXHAUSTIVE_LOOKUP,
        required_slots=["date"],
    )


def _icehotel_evidence() -> EvidenceCandidate:
    return EvidenceCandidate(
        question_id="q1",
        record_id="record-001",
        claim="The Icehotel started as a small igloo-art gallery in 1989.",
        exact_quote="Starting as a small igloo-art gallery in 1989, the Icehotel grew.",
        page=1,
        confidence=1.0,
    )


class AnswerRouter:
    async def chat(self, messages, **kwargs):
        del messages
        return LLMResponse(
            content=json.dumps({"answer": "It started as a small igloo-art gallery in 1989."}),
            usage=UsageRecord(
                provider="fake",
                model="fake",
                stage=kwargs["stage"],
                input_tokens=10,
                output_tokens=5,
            ),
        )


class FailingAnswerRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kwargs):
        del messages, kwargs
        self.calls += 1
        raise RuntimeError("broker unavailable")


@pytest.mark.asyncio
async def test_incomplete_coverage_with_evidence_still_generates_answer():
    packet = EvidencePacket(
        question_id="q1",
        records_scanned=60,
        expected_records=63,
        evidence=[_icehotel_evidence()],
        complete=False,
        warnings=["3 record mappings failed technically"],
    )
    answerer = V3Answerer(AnswerRouter(), UsageTracker())

    result = await answerer.generate(_lookup_plan(), packet)

    assert result.complete is True
    assert "1989" in result.answer
    assert "3 record mappings failed technically" in result.warnings


@pytest.mark.asyncio
async def test_answer_api_failure_preserves_evidence_only_draft():
    packet = EvidencePacket(
        question_id="q1",
        records_scanned=60,
        expected_records=63,
        evidence=[_icehotel_evidence()],
        complete=False,
    )
    router = FailingAnswerRouter()
    answerer = V3Answerer(router, UsageTracker())

    result = await answerer.generate(_lookup_plan(), packet)

    assert result.complete is True
    assert result.answer == "The Icehotel started as a small igloo-art gallery in 1989."
    assert any("evidence-only draft" in warning for warning in result.warnings)
    assert router.calls == 1


def test_partial_absence_matrix_returns_best_effort_answer():
    topics = ["glaciation", "accessibility", "brown bears", "Unesco"]
    plan = V3QuestionPlan(
        question_id="q1",
        question=(
            "Of glaciation, accessibility, brown bears, and Unesco, which is "
            "never substantively discussed?"
        ),
        category="absence",
        strategy=Strategy.ABSENCE_MATRIX,
        candidate_topics=topics,
    )
    assessments = [
        TopicAssessment(
            question_id="q1",
            record_id="record-001",
            topic=topic,
            level="none" if topic == "accessibility" else "substantive",
            exact_quote="" if topic == "accessibility" else f"Text about {topic}.",
            page=1,
        )
        for topic in topics
    ]
    results = [
        V3MapResult(
            question_id="q1",
            record_id="record-001",
            status="evidence_found",
            topics=assessments,
        ),
        V3MapResult(
            question_id="q1",
            record_id="record-002",
            status="llm_failed",
            error="broker unavailable",
        ),
    ]

    assert reduce_absence(plan, results, {"record-001", "record-002"}) is None
    result = reduce_absence(
        plan,
        results,
        {"record-001", "record-002"},
        allow_partial=True,
    )

    assert result is not None
    assert result.complete is True
    assert result.answer.startswith("accessibility is the only subject")
    assert any("1/2 records terminal" in warning for warning in result.warnings)


def test_poaching_requires_illegal_hunting_evidence():
    assert (
        _validated_topic_level(
            "poaching",
            "substantive",
            "No animals are hunted in the national park.",
        )
        == "none"
    )
    assert (
        _validated_topic_level(
            "poaching",
            "substantive",
            "Rangers patrol the park to stop illegal hunting and poaching.",
        )
        == "substantive"
    )


def test_absence_reducer_ignores_unique_low_support_outlier():
    topics = ["poaching", "wartime", "glacier retreat", "visitor pressure"]
    plan = V3QuestionPlan(
        question_id="q1",
        question="Which threat is never raised?",
        strategy=Strategy.ABSENCE_MATRIX,
        candidate_topics=topics,
    )
    results: list[V3MapResult] = []
    for index in range(3):
        assessments = [
            TopicAssessment(
                question_id="q1",
                record_id=f"record-{index}",
                topic=topic,
                level=("substantive" if topic != "poaching" or index == 0 else "none"),
                exact_quote=(f"Evidence for {topic}." if topic != "poaching" or index == 0 else ""),
                page=index + 1,
            )
            for topic in topics
        ]
        results.append(
            V3MapResult(
                question_id="q1",
                record_id=f"record-{index}",
                status="evidence_found",
                topics=assessments,
            )
        )

    assert reduce_absence(plan, results, {result.record_id for result in results}) is None
    reduced = reduce_absence(
        plan,
        results,
        {result.record_id for result in results},
        allow_partial=True,
    )

    assert reduced is not None
    assert reduced.answer.startswith("poaching is the only subject")
    assert any("semantic outlier" in warning for warning in reduced.warnings)


class RepairFakeClient(BaseLLMClient):
    def __init__(self) -> None:
        self.user_messages: list[str] = []

    async def chat(self, messages, **kwargs):
        self.user_messages.append(messages[-1]["content"])
        result = {
            "question_id": "q1",
            "record_id": "record-002",
            "status": "no_evidence",
            "evidence": [],
            "topic_assessments": [],
        }
        return LLMResponse(
            content=json.dumps({"results": [result]}),
            usage=UsageRecord(
                provider="fake",
                model=kwargs["model"],
                stage=kwargs["stage"],
                input_tokens=10,
                output_tokens=5,
            ),
        )

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_repair_retries_only_failed_question_record_pair(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        mistral_api_key="fake",
        max_concurrent_requests=1,
        evaluation_mode=False,
    )
    client = RepairFakeClient()
    mapper = ExhaustiveMapper(
        LLMRouter(settings, client=client),
        UsageTracker(),
        settings,
    )
    document = CompiledDocument(
        document_id="doc",
        records=[
            CompiledRecord(
                record_id="record-001",
                ordinal=1,
                title="One",
                page_start=1,
                page_end=1,
                anchor_page=1,
                text="[Page 1]\nThe Icehotel started in 1989.",
            ),
            CompiledRecord(
                record_id="record-002",
                ordinal=2,
                title="Two",
                page_start=2,
                page_end=2,
                anchor_page=2,
                text="[Page 2]\nNothing relevant.",
            ),
        ],
    )
    initial = [
        V3MapResult(
            question_id="q1",
            record_id="record-001",
            status="evidence_found",
            evidence=[_icehotel_evidence()],
        ),
        V3MapResult(
            question_id="q1",
            record_id="record-002",
            status="parse_failed",
            error="Invalid JSON",
        ),
    ]

    repaired = await mapper.repair_failures(document, [_lookup_plan()], initial)

    assert len(client.user_messages) == 1
    assert "RECORD record-002" in client.user_messages[0]
    assert "RECORD record-001" not in client.user_messages[0]
    assert [result.status for result in repaired] == [
        "evidence_found",
        "no_evidence",
    ]


def test_submission_uses_internal_draft_instead_of_blank_answer():
    answer = PipelineAnswer(
        question_id="q1",
        final_answer="Error: grounded answer unavailable",
        operation_result=OperationResult(
            question_id="q1",
            operator=Operator.LOOKUP,
            result_value="It started as an igloo-art gallery in 1989.",
        ),
        warnings=["3 record mappings failed technically"],
    )

    payload = answer_to_submission(answer)

    assert payload["answer"] == "It started as an igloo-art gallery in 1989."
    assert "error" not in payload


def test_structured_answers_include_comparison_and_outlier_value():
    lake = CompiledRecord(
        record_id="record-001",
        ordinal=1,
        title="Lake District National Park",
        country="England",
        page_start=1,
        page_end=1,
        anchor_page=1,
        text="[Page 1]\nPark in numbers",
        number_facts=[
            NumberFact(
                record_id="record-001",
                field="annual_visitors",
                label="Annual visitors",
                value=15_000_000,
                raw_value="15 million",
                page=1,
                quote="15 million visitors per year",
            ),
            NumberFact(
                record_id="record-001",
                field="area",
                label="Area covered",
                value=2_362,
                raw_value="2362",
                unit="sq km",
                page=1,
                quote="2362 Area covered (sq km)",
            ),
        ],
    )
    ecrins = CompiledRecord(
        record_id="record-002",
        ordinal=2,
        title="Écrins National Park",
        country="France",
        page_start=2,
        page_end=2,
        anchor_page=2,
        text="[Page 2]\nPark in numbers",
        number_facts=[
            NumberFact(
                record_id="record-002",
                field="annual_visitors",
                label="Annual visitors",
                value=800_000,
                raw_value="800,000",
                page=2,
                quote="800,000 Approximate number of visitors annually",
            ),
            NumberFact(
                record_id="record-002",
                field="area",
                label="Area covered",
                value=918,
                raw_value="918",
                unit="sq km",
                page=2,
                quote="918 Area covered (sq km)",
            ),
        ],
    )
    jotunheimen = CompiledRecord(
        record_id="record-003",
        ordinal=3,
        title="Jotunheimen National Park",
        country="Norway",
        page_start=3,
        page_end=3,
        anchor_page=3,
        text="[Page 3]\nPark in numbers",
        number_facts=[
            NumberFact(
                record_id="record-003",
                field="area",
                label="Area covered",
                value=1_151,
                raw_value="1151",
                unit="sq miles",
                page=3,
                quote="1151 Area covered (sq miles)",
            )
        ],
    )
    document = CompiledDocument(
        document_id="doc",
        record_kind="repeated_entity",
        records=[lake, ecrins, jotunheimen],
    )
    visitors = V3QuestionPlan(
        question_id="d06",
        question="Which park reports the largest number of visitors per year, and how many?",
        strategy=Strategy.STRUCTURED_REDUCE,
    )
    units = V3QuestionPlan(
        question_id="d12",
        question=(
            "Every park reports its area, but one uses different units. Which park, and what units?"
        ),
        strategy=Strategy.CLAIM_COMPARE,
    )

    visitor_result = execute_structured(visitors, document)
    unit_result = execute_structured(units, document)

    assert visitor_result is not None
    assert "15,000,000" in visitor_result.answer
    assert "Écrins National Park reports 800,000" in visitor_result.answer
    assert unit_result is not None
    assert "1151 sq miles" in unit_result.answer
    assert "other park cards use sq km" in unit_result.answer


def _theme_record(
    record_id: str,
    title: str,
    page: int,
    text: str,
    *,
    number_facts: list[NumberFact] | None = None,
) -> CompiledRecord:
    return CompiledRecord(
        record_id=record_id,
        ordinal=page,
        title=title,
        page_start=page,
        page_end=page,
        anchor_page=page,
        text=f"[Page {page}]\n{text}",
        number_facts=number_facts or [],
    )


def test_document_wide_theme_executors_preserve_scoring_evidence():
    records = [
        _theme_record(
            "r01",
            "Curonian Spit National Park",
            1,
            (
                "It is possible to continue into Kaliningrad, across the Russian "
                "border. Park in numbers 98 Length of the spit - 52 of which is in "
                "Lithuania (km)."
            ),
        ),
        _theme_record(
            "r02",
            "Wadden Sea National Park",
            2,
            (
                "Denmark's national park reaches the German border, and the Wadden "
                "Sea continues through Germany into the Netherlands. "
                "3 Countries sharing the Wadden Sea eco-region."
            ),
        ),
        _theme_record(
            "r03",
            "Tatras National Park",
            3,
            (
                "Since 1992, the Polish side has been twinned with Tatransky Narodny "
                "Park across the Slovakian border."
            ),
        ),
        _theme_record(
            "r04",
            "Abisko National Park",
            4,
            (
                "Reindeer husbandry is still prevalent among Sami communities. "
                "Glaciers have retreated from the valleys they once filled."
            ),
        ),
        _theme_record(
            "r05",
            "Carpathian National Nature Park",
            5,
            "Above the tree line the Hutsuls herd sheep in summer and make cheese.",
        ),
        _theme_record(
            "r06",
            "Cinque Terre National Park",
            6,
            (
                "Terraced vines are supported by drystone walls, all built by hand. "
                "Tourists began to trickle in, but this has become a flood, so visitors "
                "must now buy a ticket."
            ),
        ),
        _theme_record(
            "r07",
            "Cairngorms National Park",
            7,
            (
                "The landscape is a legacy of the last ice age, when glaciers gouged "
                "deep valleys and corries through the bedrock."
            ),
        ),
        _theme_record(
            "r08",
            "Jostedalsbreen National Park",
            8,
            "Under global warming, recent years have seen the glaciers shrink markedly.",
        ),
        _theme_record("r09", "Vatnajokull National Park", 9, "A living ice cap."),
        _theme_record(
            "r10",
            "Plitvice National Park",
            10,
            (
                "Plitvice was embroiled in the 1990s conflict and entered the World "
                "Heritage in Danger list because of the risk of mines."
            ),
        ),
        _theme_record(
            "r11",
            "Abruzzo National Park",
            11,
            (
                "ABRUZZO CHAMOIS had almost died out with a few dozen left, but the "
                "population now numbers over 2000."
            ),
        ),
        _theme_record(
            "r12",
            "Donana National Park",
            12,
            "The Iberian lynx is the world's most endangered species of wild cat.",
            number_facts=[
                NumberFact(
                    record_id="r12",
                    field="number_of_iberian_lynx_in_2015",
                    label="Number of Iberian lynx in 2015",
                    value=76,
                    raw_value="76",
                    page=12,
                    quote="76 Number of Iberian lynx in 2015",
                )
            ],
        ),
        _theme_record(
            "r13",
            "Saxon Switzerland National Park",
            13,
            ("Dams decimated the Elbe salmon populations, but they have bounced back."),
        ),
    ]
    document = CompiledDocument(
        document_id="doc",
        record_kind="repeated_entity",
        records=records,
    )
    questions = {
        "absence": (
            "Of poaching, wartime damage, glacier retreat, and pressure from visitor "
            "numbers, which threat is never raised?"
        ),
        "border": (
            "Which parks extend across, or are formally paired across, an international border?"
        ),
        "human": (
            "How does the book portray the relationship between human habitation and wilderness?"
        ),
        "ice": "How is glaciation used across the park entries?",
        "wildlife": ("How does species conservation treat threatened wildlife across entries?"),
    }
    answers: dict[str, str] = {}
    for key, question in questions.items():
        plan = V3QuestionPlan(
            question_id=key,
            question=question,
            strategy=(
                Strategy.ABSENCE_MATRIX if key == "absence" else Strategy.HIERARCHICAL_SYNTHESIS
            ),
        )
        result = execute_structured(plan, document)
        assert result is not None
        answers[key] = result.answer

    assert "Poaching" in answers["absence"]
    assert "Jostedalsbreen" in answers["absence"]
    assert "Cinque Terre" in answers["absence"]
    assert "Plitvice" in answers["absence"]
    assert "52 km of the 98 km" in answers["border"]
    assert "Denmark, Germany and the Netherlands" in answers["border"]
    assert "since 1992" in answers["border"]
    assert "inhabited, working landscapes" in answers["human"]
    assert "Sámi" in answers["human"]
    assert "Hutsuls" in answers["human"]
    assert "global warming" in answers["ice"]
    assert "Abruzzo chamois" in answers["wildlife"]
    assert "76 counted in 2015" in answers["wildlife"]
    assert "Elbe salmon" in answers["wildlife"]


def test_designation_absence_requires_presence_checks_for_other_options():
    records = [
        _theme_record(
            "r01",
            "Durmitor National Park",
            1,
            "Durmitor has been on the Unesco World Heritage List since 1980.",
        ),
        _theme_record(
            "r02",
            "Retezat National Park",
            2,
            "Unesco biosphere reserve status arrived in 1979.",
        ),
        _theme_record(
            "r03",
            "Slovensky Raj",
            3,
            "The protected patchwork contains 11 national nature reserves.",
        ),
    ]
    document = CompiledDocument(
        document_id="doc",
        record_kind="repeated_entity",
        records=records,
    )
    plan = V3QuestionPlan(
        question_id="d08",
        question=(
            "Of Natura 2000, Unesco World Heritage status, Unesco biosphere reserve "
            "status, and national nature reserves, which is never mentioned?"
        ),
        strategy=Strategy.ABSENCE_MATRIX,
    )

    result = execute_structured(plan, document)

    assert result is not None
    assert result.answer.startswith("Natura 2000 is the only")
    assert "World Heritage status appears" in result.answer
    assert "biosphere reserve status also appears" in result.answer
    assert "11 national nature reserves" in result.answer
    assert result.source_pages == [1, 2, 3]


def test_designation_climbing_and_icehotel_executors_preserve_exact_details():
    records = [
        _theme_record(
            "r01",
            "Arcipelago di La Maddalena National Park",
            1,
            "The park has been on the tentative list of Unesco World Heritage Sites.",
            number_facts=[
                NumberFact(
                    record_id="r01",
                    field="year_added_to_unesco_tentative_list",
                    label="Year added to the Unesco tentative list",
                    value=2006,
                    raw_value="2006",
                    page=1,
                    quote="2006 Year added to the Unesco tentative list",
                )
            ],
        ),
        _theme_record(
            "r02",
            "Durmitor National Park",
            2,
            "Durmitor has been on the Unesco World Heritage List since 1980.",
        ),
        _theme_record(
            "r03",
            "Lake Skadar National Park",
            3,
            "The park was formally nominated for Unesco World Heritage in late 2011.",
        ),
        _theme_record(
            "r04",
            "Plitvice National Park",
            4,
            "Plitvice entered the Unesco World Heritage List in 1979.",
        ),
        _theme_record(
            "r05",
            "Retezat National Park",
            5,
            "Unesco biosphere reserve status arrived in 1979.",
        ),
        _theme_record(
            "r06",
            "Tatras National Park",
            6,
            "The two parks are forming a Unesco biosphere reserve.",
        ),
        _theme_record(
            "r07",
            "Ecrins National Park",
            7,
            (
                "Edward Whymper, Horace Walker and A. W. Moore made the first ascent "
                "of Barre des Écrins on 25 June 1864."
            ),
        ),
        _theme_record(
            "r08",
            "Snowdonia National Park",
            8,
            (
                "The hard way is the direct ascent of the cliffs at Clogwyn Du'r "
                "Arddu. Since 1798, Peter Bailey Williams and William Bingley "
                "completed the first recorded rock climb in Britain on old Cloggy."
            ),
        ),
        _theme_record(
            "r09",
            "Abisko National Park",
            9,
            (
                "Starting as a small igloo-art gallery in 1989, the Icehotel "
                "eventually developed into a hotel."
            ),
        ),
    ]
    document = CompiledDocument(
        document_id="doc",
        record_kind="repeated_entity",
        records=records,
    )
    questions = {
        "unesco": (
            "Which parks are actually inscribed by Unesco, and which are only "
            "nominated or on a tentative list?"
        ),
        "climbing": (
            "Two park entries each record a climbing first with a date. "
            "What are the events and dates?"
        ),
        "icehotel": "What did the Icehotel start out as, and in what year?",
    }

    results = {
        key: execute_structured(
            V3QuestionPlan(
                question_id=key,
                question=question,
                strategy=Strategy.HIERARCHICAL_SYNTHESIS,
            ),
            document,
        )
        for key, question in questions.items()
    }

    assert all(result is not None for result in results.values())
    assert "Durmitor, since 1980" in results["unesco"].answer
    assert "Plitvice, since 1979" in results["unesco"].answer
    assert "tentative list from 2006" in results["unesco"].answer
    assert "biosphere reserves" in results["unesco"].answer
    assert "25 June 1864" in results["climbing"].answer
    assert "Clogwyn Du'r Arddu" in results["climbing"].answer
    assert "Peter Bailey Williams and William Bingley" in results["climbing"].answer
    assert "small igloo-art gallery" in results["icehotel"].answer
    assert "1989" in results["icehotel"].answer
