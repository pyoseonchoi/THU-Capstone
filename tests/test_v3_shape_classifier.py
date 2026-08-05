"""Routing questions to operations by meaning rather than by phrasing."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.schemas import QuestionRequest
from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    NumberFact,
    OperationKind,
    QuestionShape,
    ShapePlan,
)
from app.v3.question_compiler import compile_question
from app.v3.shape_classifier import ShapeClassifier
from app.v3.structured_executor import execute_structured


class ScriptedClient(BaseLLMClient):
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls = 0

    async def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        self.last_prompt = messages[-1]["content"]
        return LLMResponse(content=json.dumps(self._payload))

    async def close(self) -> None:
        return None


def _record(ordinal: int, title: str, group: str, area: float) -> CompiledRecord:
    record_id = f"record-{ordinal:03d}"
    return CompiledRecord(
        record_id=record_id,
        ordinal=ordinal,
        title=title,
        country=group,
        page_start=ordinal,
        page_end=ordinal,
        anchor_page=ordinal,
        text=f"[Page {ordinal}]\n{title}",
        number_facts=[
            NumberFact(
                record_id=record_id,
                field="area_covered",
                label="Area covered (sq km)",
                value=area,
                raw_value=str(area),
                unit="sq km",
                page=ordinal,
                quote=f"Area covered (sq km): {area}",
            )
        ],
    )


def _document() -> CompiledDocument:
    return CompiledDocument(
        document_id="d",
        record_kind="repeated_entity",
        entity_label="park",
        records=[
            _record(1, "Alpha Park", "Croatia", 100),
            _record(2, "Beta Park", "Croatia", 300),
            _record(3, "Gamma Park", "Montenegro", 200),
            _record(4, "Delta Park", "Norway", 150),
        ],
        field_catalog=["area_covered"],
        registry_trusted=True,
    )


def _classifier(tmp_path, payload):
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:1/v1",
        data_dir=tmp_path,
    )
    client = ScriptedClient(payload)
    return ShapeClassifier(LLMRouter(settings, client=client), UsageTracker(), settings), client


@pytest.mark.asyncio
async def test_routes_a_question_whose_wording_no_gate_would_match(tmp_path):
    # None of the retired phrasings appear; only the meaning does.
    classifier, client = _classifier(tmp_path, {
        "questions": [{
            "question_id": "q1",
            "shape": "extremum",
            "field": "area_covered",
            "direction": "max",
            "confident": True,
        }]
    })
    document = _document()
    question = QuestionRequest(
        question_id="q1",
        question="Which of these sites sprawls across the most ground?",
        category="superlative",
    )

    shapes = await classifier.classify([question], document)
    plan = compile_question(question, document, None, shapes["q1"])
    result = execute_structured(plan, document)

    assert plan.target_fields == ["area_covered"]
    assert [step.kind for step in plan.operations] == [OperationKind.ARGMAX]
    assert result is not None
    assert "Beta Park" in result.answer
    # The router is told what the document holds, not what it is about.
    assert "area_covered" in client.last_prompt
    assert "Croatia" in client.last_prompt


@pytest.mark.asyncio
async def test_a_field_the_document_lacks_is_refused(tmp_path):
    classifier, _ = _classifier(tmp_path, {
        "questions": [{
            "question_id": "q1",
            "shape": "extremum",
            "field": "invented_column",
            "direction": "max",
            "confident": True,
        }]
    })

    shapes = await classifier.classify(
        [QuestionRequest(question_id="q1", question="Which is biggest?", category="superlative")],
        _document(),
    )

    # The shape survives for the record, but without a field it cannot route.
    assert shapes["q1"].field == ""
    assert shapes["q1"].routes is False


@pytest.mark.asyncio
async def test_an_unsure_routing_does_not_drive_an_executor(tmp_path):
    classifier, _ = _classifier(tmp_path, {
        "questions": [{
            "question_id": "q1",
            "shape": "count_entities",
            "groups": ["Croatia"],
            "confident": False,
        }]
    })
    document = _document()
    question = QuestionRequest(
        question_id="q1", question="Anything about Croatia?", category="aggregation"
    )

    shapes = await classifier.classify([question], document)
    plan = compile_question(question, document, None, shapes["q1"])

    assert shapes["q1"].routes is False
    assert execute_structured(plan, document) is None


@pytest.mark.asyncio
async def test_groups_outside_the_registry_are_dropped(tmp_path):
    classifier, _ = _classifier(tmp_path, {
        "questions": [{
            "question_id": "q1",
            "shape": "count_entities",
            "groups": ["Croatia", "Atlantis"],
            "confident": True,
        }]
    })

    shapes = await classifier.classify(
        [QuestionRequest(question_id="q1", question="How many?", category="aggregation")],
        _document(),
    )

    assert shapes["q1"].groups == ["Croatia"]


@pytest.mark.asyncio
async def test_counting_by_group_answers_from_the_registry(tmp_path):
    classifier, _ = _classifier(tmp_path, {
        "questions": [{
            "question_id": "q1",
            "shape": "count_entities",
            "groups": ["Croatia", "Montenegro"],
            "confident": True,
        }]
    })
    document = _document()
    question = QuestionRequest(
        question_id="q1",
        question="Break the sites down by the nations they sit in.",
        category="aggregation",
    )

    shapes = await classifier.classify([question], document)
    result = execute_structured(
        compile_question(question, document, None, shapes["q1"]), document
    )

    assert result is not None
    assert "Croatia has 2" in result.answer
    assert "Montenegro has 1" in result.answer


@pytest.mark.asyncio
async def test_a_shape_missing_its_arguments_declines(tmp_path):
    # A threshold count without a comparator or a value cannot be executed.
    classifier, _ = _classifier(tmp_path, {
        "questions": [{
            "question_id": "q1",
            "shape": "count_by_threshold",
            "field": "area_covered",
            "confident": True,
        }]
    })

    shapes = await classifier.classify(
        [QuestionRequest(question_id="q1", question="How many are big?", category="aggregation")],
        _document(),
    )

    assert shapes["q1"].routes is False


def test_routing_wins_over_wording_for_a_question_it_shaped():
    # The question carries a retired phrasing, but the router placed it
    # elsewhere; the phrasing must not pull it back.
    document = _document()
    question = QuestionRequest(
        question_id="q1",
        question="Every park reports its area, but one uses different units. Which?",
        category="contradiction",
    )
    shaped = ShapePlan(shape=QuestionShape.SYNTHESIS, confident=True)

    plan = compile_question(question, document, None, shaped)

    assert execute_structured(plan, document) is None
