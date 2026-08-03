"""LLM router: routes requests to the correct model for each pipeline stage."""

from __future__ import annotations

from app.config import LLMProvider, Settings
from app.exceptions import LLMConfigurationError
from app.llm.base import BaseLLMClient, LLMResponse
from app.logging_config import get_logger

logger = get_logger("llm.router")


class LLMRouter:
    """Routes LLM calls to the appropriate model/client by pipeline stage.

    Stages:
        planner   → settings.planner_model
        mapper    → settings.mapper_model
        verifier  → settings.verifier_model
        answer    → settings.answer_model
    """

    def __init__(self, settings: Settings, client: BaseLLMClient | None = None):
        self._settings = settings
        self._model_map = {
            "planner": settings.planner_model,
            "mapper": settings.mapper_model,
            "verifier": settings.verifier_model,
            "answer": settings.answer_model,
        }
        self._client = client or self._create_client(settings)

    @staticmethod
    def _create_client(settings: Settings) -> BaseLLMClient:
        if settings.llm_provider == LLMProvider.MISTRAL_API:
            from app.llm.mistral_client import MistralClient
            return MistralClient(settings)
        elif settings.llm_provider in (LLMProvider.OPENAI_COMPATIBLE, LLMProvider.OLLAMA):
            from app.llm.openai_compatible_client import OpenAICompatibleClient
            return OpenAICompatibleClient(settings)
        else:
            raise LLMConfigurationError(
                f"Unknown LLM provider: {settings.llm_provider}"
            )

    def get_model(self, stage: str) -> str:
        """Return the model identifier for a pipeline stage."""
        model = self._model_map.get(stage)
        if not model:
            raise LLMConfigurationError(f"No model configured for stage: {stage}")
        return model

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        chunk_id: str = "",
        question_ids: list[str] | None = None,
    ) -> LLMResponse:
        """Route a chat request to the correct model for the stage."""
        model = self.get_model(stage)
        logger.debug("Routing stage=%s to model=%s", stage, model)
        return await self._client.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stage=stage,
            chunk_id=chunk_id,
            question_ids=question_ids,
        )

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()
