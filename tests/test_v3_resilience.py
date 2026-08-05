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
from app.v3.exhaustive_mapper import (
    ExhaustiveMapper,
    _exact_topic_quote,
    _validated_topic_level,
)
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


def test_literal_absence_declaration_is_not_reclassified_as_presence():
    record = CompiledRecord(
        record_id="closing",
        ordinal=1,
        title="Closing synthesis",
        page_start=9,
        page_end=9,
        anchor_page=9,
        text=(
            "[Page 9]\nThe guide does not substantively discuss accessibility, "
            "the GeoArk Alliance, or cyber sabotage."
        ),
    )

    assert _exact_topic_quote(record, "GeoArk Alliance") is None

    record.text += " A GeoArk Alliance project coordinates measurements across borders."
    match = _exact_topic_quote(record, "GeoArk Alliance")
    assert match is not None
    assert "coordinates measurements" in match[0]


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


def test_absence_reducer_never_ignores_positive_support():
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

    assert reduced is None


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

    # Only two of the three records bind a visitor figure, so the largest one
    # cannot be proven from the compiled table: the unbound record may hold a
    # bigger number. The deterministic path must decline and let the
    # exhaustive mapper read every record instead of answering from the rows
    # that happened to parse.
    assert visitor_result is None
    # Area is bound for every record, so the unit outlier stays deterministic.
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


def test_generic_needle_executor_preserves_exact_details():
    records = [
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
    plan = V3QuestionPlan(
        question_id="icehotel",
        question="What did the Icehotel start out as, and in what year?",
        strategy=Strategy.EXHAUSTIVE_LOOKUP,
    )
    result = execute_structured(plan, document)
    assert result is not None
    assert "small igloo-art gallery" in result.answer
    assert "1989" in result.answer
