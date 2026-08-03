"""Course broker-specific monitoring helpers."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import LLMProvider, Settings
from app.exceptions import LLMConfigurationError, LLMRequestError


async def fetch_broker_usage(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch the authenticated group's metered broker usage."""
    if settings.llm_provider != LLMProvider.OPENAI_COMPATIBLE:
        raise LLMConfigurationError(
            "Broker usage is available only for openai_compatible provider"
        )
    if not settings.mistral_api_key:
        raise LLMConfigurationError("MISTRAL_API_KEY is not configured")

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    try:
        response = await http.get(
            f"{settings.effective_base_url.rstrip('/')}/usage",
            headers={
                "Authorization": f"Bearer {settings.mistral_api_key}",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Broker returned a non-object usage response")
        return payload
    except (httpx.HTTPError, ValueError) as exc:
        raise LLMRequestError(f"Broker usage request failed: {exc}") from exc
    finally:
        if owns_client:
            await http.aclose()
