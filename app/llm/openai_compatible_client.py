"""OpenAI-compatible client for university / local endpoints serving Mistral models.

This client must ONLY be used to call one of the two approved Mistral model
families (Mistral Small 3.2, Ministral 3 8B). It must never be used with
unapproved models.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.exceptions import LLMConfigurationError, LLMRequestError
from app.llm.base import BaseLLMClient, LLMResponse
from app.logging_config import get_logger
from app.schemas import UsageRecord

logger = get_logger("llm.openai_compatible")


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible endpoints serving approved Mistral models."""

    def __init__(self, settings: Settings):
        base_url = settings.effective_base_url
        if not base_url:
            raise LLMConfigurationError(
                "LLM_BASE_URL or OLLAMA_BASE_URL is required for openai_compatible/ollama provider."
            )
        self._base_url = base_url.rstrip("/")
        self._settings = settings
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers=self._build_headers(settings),
        )
        self._prompt_versions = {
            "planner": settings.prompt_version_planner,
            "mapper": settings.prompt_version_mapper,
            "verifier": settings.prompt_version_verifier,
            "answer": settings.prompt_version_answer,
        }

    @staticmethod
    def _build_headers(settings: Settings) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.mistral_api_key:
            headers["Authorization"] = f"Bearer {settings.mistral_api_key}"
        return headers

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stage: str = "",
        chunk_id: str = "",
        question_ids: list[str] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the OpenAI-compatible endpoint."""
        start = time.perf_counter()
        retry_count = 0

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/chat/completions"

        @retry(
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            stop=stop_after_attempt(self._settings.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            before_sleep=lambda rs: logger.warning(
                "Retry %d for stage=%s model=%s", rs.attempt_number, stage, model
            ),
            reraise=True,
        )
        async def _call():
            nonlocal retry_count
            try:
                resp = await self._http.post(url, json=body)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                retry_count += 1
                logger.error("OpenAI-compat API error (stage=%s): %s", stage, exc)
                raise

        try:
            data = await _call()
        except httpx.ConnectError as exc:
            message = f"Cannot connect to the LLM endpoint at {self._base_url}."
            if self._settings.llm_provider.value == "ollama":
                message += " Start Ollama with 'ollama serve' and try again."
            raise LLMRequestError(
                message,
                model=model,
                retry_count=retry_count,
            ) from exc
        except Exception as exc:
            raise LLMRequestError(
                f"OpenAI-compatible API call failed: {exc}",
                model=model,
                retry_count=retry_count,
            ) from exc

        latency = (time.perf_counter() - start) * 1000

        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens")
        output_tokens = usage_data.get("completion_tokens")

        usage = UsageRecord(
            provider="openai_compatible",
            model=model,
            stage=stage,
            prompt_version=self._prompt_versions.get(stage, ""),
            chunk_id=chunk_id,
            question_ids=question_ids or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            retry_count=retry_count,
            success=True,
        )

        return LLMResponse(content=content, usage=usage, raw=data)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
