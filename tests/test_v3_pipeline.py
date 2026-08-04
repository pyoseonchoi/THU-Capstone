"""End-to-end tests for the single V3 runtime path."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.pipeline import FullScanPipeline
from app.schemas import QuestionRequest, UsageRecord
from app.v3.compiler import COMPILER_VERSION


class PipelineFakeClient(BaseLLMClient):
    def __init__(self):
        self.stages: list[str] = []

    async def chat(self, messages, **kwargs):
        self.stages.append(kwargs["stage"])
        if kwargs["stage"] == "mapper":
            user = messages[-1]["content"]
            record_id = "segment-001"
            assert record_id in user
            content = {
                "results": [{
                    "question_id": "q1",
                    "record_id": record_id,
                    "status": "evidence_found",
                    "evidence": [{
                        "claim": "It was called Aurora.",
                        "exact_quote": "The internal code name was Aurora.",
                        "page": 1,
                        "entity": "Aurora",
                        "field": "code_name",
                        "confidence": 1.0,
                    }],
                    "topic_assessments": [],
                }],
            }
        else:
            content = {"answer": "The internal code name was Aurora."}
        return LLMResponse(
            content=json.dumps(content),
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
async def test_unresolved_question_maps_reduces_and_answers(tmp_path):
    source = tmp_path / "document.txt"
    source.write_text("The internal code name was Aurora.", encoding="utf-8")
    fake = PipelineFakeClient()
    settings = Settings(
        data_dir=tmp_path / "data",
        mistral_api_key="fake",
        max_concurrent_requests=1,
        evaluation_mode=False,
    )
    pipeline = FullScanPipeline(settings, llm_client=fake)
    metadata, chunks = await pipeline.process_document(source)
    run = await pipeline.answer_questions(
        metadata.document_id,
        [QuestionRequest(
            question_id="q1",
            question="What was the internal code name?",
            category="needle",
        )],
    )

    assert len(chunks) == 1
    assert run.pipeline_mode == "ADAPTIVE_HIERARCHICAL_V3"
    assert run.answers[0].final_answer == "The internal code name was Aurora."
    assert run.answers[0].coverage.coverage_percentage == 100
    assert fake.stages == ["mapper", "answer"]
    await pipeline.close()


@pytest.mark.asyncio
async def test_structured_question_uses_no_llm_calls(tmp_path):
    source = tmp_path / "parks.txt"
    pages = []
    for page, title, area in (
        (1, "Alpha National Park", 100),
        (2, "Beta National Park", 300),
        (3, "Gamma National Park", 200),
    ):
        pages.append(
            f"<!-- page-start-marker-{page} -->\n"
            f"# {title}\nCountry: Spain\nPark in numbers\n"
            f"{area}\nArea covered (sq km)\n"
        )
    source.write_text("\n".join(pages), encoding="utf-8")
    fake = PipelineFakeClient()
    settings = Settings(
        data_dir=tmp_path / "data",
        mistral_api_key="fake",
        evaluation_mode=False,
    )
    pipeline = FullScanPipeline(settings, llm_client=fake)
    metadata, _ = await pipeline.process_document(source)
    run = await pipeline.answer_questions(
        metadata.document_id,
        [QuestionRequest(
            question_id="q1",
            question="Of all the parks, which covers the largest area, and how large is it?",
            category="superlative",
        )],
    )

    assert "Beta National Park" in run.answers[0].final_answer
    assert "300" in run.answers[0].final_answer
    assert fake.stages == []
    await pipeline.close()


@pytest.mark.asyncio
async def test_stale_compiled_document_is_rebuilt_from_parsed_pages(tmp_path):
    source = tmp_path / "document.txt"
    source.write_text("A small document.", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "data",
        mistral_api_key="fake",
        evaluation_mode=False,
    )
    pipeline = FullScanPipeline(settings, llm_client=PipelineFakeClient())
    metadata, _ = await pipeline.process_document(source)
    stale = pipeline._store.load_compiled_document(metadata.document_id)
    assert stale is not None
    stale.compiler_version = "v3.0"
    pipeline._store.save_compiled_document(stale)

    rebuilt = pipeline._load_or_compile(metadata.document_id)

    assert rebuilt is not None
    assert rebuilt.compiler_version == COMPILER_VERSION
    assert pipeline._store.load_compiled_document(
        metadata.document_id
    ).compiler_version == COMPILER_VERSION
    await pipeline.close()
