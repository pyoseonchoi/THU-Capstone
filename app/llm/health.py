"""Fast connectivity checks for configured LLM providers."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import LLMProvider, Settings


async def check_llm_health(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Check endpoint reachability and configured model availability."""
    if settings.llm_provider == LLMProvider.MISTRAL_API:
        available = bool(settings.mistral_api_key)
        return {
            "available": available,
            "provider": settings.llm_provider.value,
            "message": (
                "Mistral API is configured."
                if available
                else "MISTRAL_API_KEY is not configured."
            ),
            "missing_models": [],
        }

    base_url = settings.effective_base_url
    if not base_url:
        return {
            "available": False,
            "provider": settings.llm_provider.value,
            "message": "No LLM endpoint URL is configured.",
            "missing_models": [],
        }

    models_url = f"{base_url.rstrip('/')}/models"
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(3.0))
    try:
        headers = {}
        if settings.mistral_api_key:
            headers["Authorization"] = f"Bearer {settings.mistral_api_key}"
        response = await http.get(models_url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        available_models = {
            str(item.get("id", item.get("name", "")))
            for item in payload.get("data", payload.get("models", []))
            if isinstance(item, dict)
        }
        configured_models = {
            settings.planner_model,
            settings.mapper_model,
            settings.answer_model,
            settings.verifier_model,
        }
        missing = sorted(configured_models - available_models)
        return {
            "available": not missing,
            "provider": settings.llm_provider.value,
            "message": (
                "LLM endpoint is ready."
                if not missing
                else f"Configured models are unavailable: {', '.join(missing)}"
            ),
            "missing_models": missing,
        }
    except httpx.ConnectError:
        message = f"Cannot connect to the LLM endpoint at {base_url}."
        if settings.llm_provider == LLMProvider.OLLAMA:
            message += " Start Ollama with 'ollama serve' and try again."
        return {
            "available": False,
            "provider": settings.llm_provider.value,
            "message": message,
            "missing_models": [],
        }
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "available": False,
            "provider": settings.llm_provider.value,
            "message": f"LLM health check failed: {exc}",
            "missing_models": [],
        }
    finally:
        if owns_client:
            await http.aclose()
