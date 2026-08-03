"""Tests for fast LLM endpoint health checks."""

from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.llm.health import check_llm_health


@pytest.mark.asyncio
async def test_ollama_health_reports_available_models():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "qwen2.5:7b"}]})

    settings = Settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        planner_model="qwen2.5:7b",
        mapper_model="qwen2.5:7b",
        answer_model="qwen2.5:7b",
        verifier_model="qwen2.5:7b",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        status = await check_llm_health(settings, client=client)

    assert status["available"] is True
    assert status["missing_models"] == []


@pytest.mark.asyncio
async def test_openai_compatible_health_sends_bearer_token():
    seen_authorization = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_authorization
        seen_authorization = request.headers.get("Authorization", "")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "mistral-small-3-2"},
                    {"id": "ministral-3b-2512"},
                ]
            },
        )

    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://broker.test/v1",
        mistral_api_key="group-key",
        planner_model="mistral-small-3-2",
        mapper_model="ministral-3b-2512",
        answer_model="mistral-small-3-2",
        verifier_model="mistral-small-3-2",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        status = await check_llm_health(settings, client=client)

    assert status["available"] is True
    assert seen_authorization == "Bearer group-key"


@pytest.mark.asyncio
async def test_ollama_health_explains_connection_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    settings = Settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        status = await check_llm_health(settings, client=client)

    assert status["available"] is False
    assert "ollama serve" in status["message"]
