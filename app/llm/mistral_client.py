"""Mistral API client using the official mistralai SDK."""

from __future__ import annotations

import time
from typing import Any

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

logger = get_logger("llm.mistral_client")


class MistralClient(BaseLLMClient):
    """Client for the official Mistral API via mistralai SDK."""

    def __init__(self, settings: Settings):
        if not settings.mistral_api_key:
            raise LLMConfigurationError(
                "MISTRAL_API_KEY is required for mistral_api provider."
            )
        try:
            from mistralai.client import Mistral
        except ImportError as exc:
            raise LLMConfigurationError(
                "mistralai package is required. Install with: pip install mistralai"
            ) from exc

        self._client = Mistral(api_key=settings.mistral_api_key)
        self._settings = settings
        self._prompt_versions = {
            "planner": settings.prompt_version_planner,
            "mapper": settings.prompt_version_mapper,
            "verifier": settings.prompt_version_verifier,
            "answer": settings.prompt_version_answer,
        }

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
        """Send a chat request via the Mistral SDK."""
        start = time.perf_counter()
        retry_count = 0

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self._settings.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            before_sleep=lambda rs: logger.warning(
                "Retry %d for stage=%s model=%s", rs.attempt_number, stage, model
            ),
        )
        async def _call():
            nonlocal retry_count
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = await self._client.chat.complete_async(**kwargs)
                return resp
            except Exception as exc:
                retry_count += 1
                logger.error("Mistral API error (stage=%s): %s", stage, exc)
                raise

        try:
            resp = await _call()
        except Exception as exc:
            raise LLMRequestError(
                f"Mistral API call failed after retries: {exc}",
                model=model,
                retry_count=retry_count,
            ) from exc

        latency = (time.perf_counter() - start) * 1000

        content = ""
        if resp.choices:
            content = resp.choices[0].message.content or ""

        # Extract usage
        input_tokens = None
        output_tokens = None
        if resp.usage:
            input_tokens = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens

        usage = UsageRecord(
            provider="mistral_api",
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

        return LLMResponse(content=content, usage=usage)

    async def close(self) -> None:
        """Close the Mistral client."""
        pass  # SDK handles cleanup
