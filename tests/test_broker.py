"""Tests for course broker usage integration."""

from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.llm.broker import fetch_broker_usage


@pytest.mark.asyncio
async def test_fetch_broker_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer group-key"
        return httpx.Response(200, json={
            "group": "team",
            "models": {},
            "total": {
                "requests": 2,
                "input_tokens": 100,
                "output_tokens": 20,
                "cost": 0.01,
            },
        })

    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://broker.test/v1",
        mistral_api_key="group-key",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        usage = await fetch_broker_usage(settings, client=client)

    assert usage["total"]["requests"] == 2
