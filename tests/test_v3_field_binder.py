"""Binding questions to compiled fields by meaning."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.schemas import QuestionRequest
from app.v3.field_binder import FieldBinder, field_descriptions
from app.v3.models import CompiledDocument, CompiledRecord, NumberFact


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


def _document() -> CompiledDocument:
    records = []
    for index in range(1, 5):
        record_id = f"record-{index:03d}"
        records.append(CompiledRecord(
            record_id=record_id,
            ordinal=index,
            title=f"Station {index}",
            page_start=index,
            page_end=index,
            anchor_page=index,
            text=f"[Page {index}]",
            number_facts=[
                NumberFact(
                    record_id=record_id,
                    field="annual_visiting_researchers",
                    label="Annual visiting researchers",
                    value=1000 * index,
                    raw_value=str(1000 * index),
                    page=index,
                    quote=f"Annual visiting researchers {1000 * index}",
                ),
            ],
        ))
    return CompiledDocument(
        document_id="d",
        record_kind="repeated_entity",
        records=records,
        field_catalog=["annual_visiting_researchers"],
    )


def _binder(tmp_path, payload):
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:1/v1",
        data_dir=tmp_path,
    )
    client = ScriptedClient(payload)
    return FieldBinder(LLMRouter(settings, client=client), UsageTracker(), settings), client


@pytest.mark.asyncio
async def test_binds_question_to_field_named_differently(tmp_path):
    binder, client = _binder(tmp_path, {
        "bindings": [{"question_id": "q1", "field": "annual_visiting_researchers"}]
    })
    questions = [QuestionRequest(
        question_id="q1",
        question="Which station reports the most visitors per year?",
        category="superlative",
    )]

    bound = await binder.bind(questions, _document())

    assert bound == {"q1": "annual_visiting_researchers"}
    # The document's own wording is what lets the model bridge the synonym.
    assert "Annual visiting researchers" in client.last_prompt


@pytest.mark.asyncio
async def test_field_outside_the_catalog_is_refused(tmp_path):
    # A field the document does not have would answer a different question.
    binder, _ = _binder(tmp_path, {
        "bindings": [{"question_id": "q1", "field": "invented_column"}]
    })
    questions = [QuestionRequest(
        question_id="q1", question="Which is biggest?", category="superlative"
    )]

    assert await binder.bind(questions, _document()) == {}


@pytest.mark.asyncio
async def test_unbound_question_is_omitted(tmp_path):
    binder, _ = _binder(tmp_path, {"bindings": [{"question_id": "q1", "field": ""}]})
    questions = [QuestionRequest(
        question_id="q1", question="Which name is prettiest?", category="superlative"
    )]

    assert await binder.bind(questions, _document()) == {}


def test_thinly_reported_fields_are_not_offered():
    document = _document()
    document.records[0].number_facts.append(NumberFact(
        record_id="record-001",
        field="dragons",
        label="Dragons",
        value=3,
        raw_value="3",
        page=1,
        quote="Dragons 3",
    ))
    document.field_catalog = ["annual_visiting_researchers", "dragons"]

    offered = {field for field, _, _ in field_descriptions(document)}

    assert offered == {"annual_visiting_researchers"}
