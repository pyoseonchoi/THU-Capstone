"""Abstract base for LLM clients."""

from __future__ import annotations

import abc
from typing import Any, Optional

from pydantic import BaseModel

from app.schemas import UsageRecord


class LLMResponse(BaseModel):
    """Standardised response from any LLM client."""
    content: str = ""
    usage: Optional[UsageRecord] = None
    raw: dict[str, Any] = {}


class BaseLLMClient(abc.ABC):
    """Interface that every LLM provider must implement."""

    @abc.abstractmethod
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
        """Send a chat completion request.

        Args:
            messages: List of {role, content} dicts.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            json_mode: If True, request JSON output.
            stage: Pipeline stage name for tracking.
            chunk_id: Chunk ID for tracking.
            question_ids: Question IDs being processed.

        Returns:
            LLMResponse with content and usage.
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...
