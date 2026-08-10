"""Application configuration loaded from environment variables."""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

BROKER_MODEL_IDS = {
    "mistral-small-3-2",
    "ministral-3b-2512",
}


class LLMProvider(str, enum.Enum):
    MISTRAL_API = "mistral_api"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


class PipelineMode(str, enum.Enum):
    ADAPTIVE_HIERARCHICAL = "ADAPTIVE_HIERARCHICAL"


class Settings(BaseSettings):
    """All configuration sourced from environment / .env file."""

    # LLM provider
    llm_provider: LLMProvider = LLMProvider.MISTRAL_API
    mistral_api_key: str = ""
    llm_base_url: str = ""
    ollama_base_url: str = ""

    # Model identifiers – adjust to match your endpoint
    planner_model: str = "mistral-small-3-2"
    mapper_model: str = "ministral-3b-2512"
    verifier_model: str = "mistral-small-3-2"
    answer_model: str = "mistral-small-3-2"

    # Concurrency & reliability
    max_concurrent_requests: int = Field(default=4, ge=1, le=64)
    request_timeout_seconds: int = Field(default=120, ge=10)
    max_retries: int = Field(default=4, ge=0)

    # Chunking
    chunk_target_tokens: int = Field(default=3500, ge=500)
    chunk_overlap_tokens: int = Field(default=250, ge=0)

    # Question batching
    question_batch_size: int = Field(default=4, ge=1, le=10)
    record_batch_size: int = Field(default=3, ge=1, le=6)
    record_batch_max_characters: int = Field(default=18_000, ge=4_000, le=100_000)

    # Storage
    data_dir: Path = Path("./data")

    # Vision fallback for low-text pages
    enable_vision_fallback: bool = False

    # Per-record structured profiling. Deterministic reduction needs a fact
    # table the layout parser alone cannot fill on unfamiliar documents.
    # Questions are matched to compiled fields by meaning, not by wording.
    enable_field_binding: bool = True
    # Questions are routed to an operation by meaning, not by phrasing.
    enable_shape_classification: bool = True
    enable_record_profiling: bool = True
    profile_min_coverage: float = Field(default=0.3, ge=0.0, le=1.0)
    profile_vocabulary_sample: int = Field(default=6, ge=0, le=40)

    # Pipeline mode
    pipeline_mode: PipelineMode = PipelineMode.ADAPTIVE_HIERARCHICAL
    evaluation_mode: bool = False

    # Submission metadata
    team_name: str = ""
    submission_notes: str = (
        "Schema-compiled entity and table catalogs, caption-validated contents indexes, "
        "composable Python reduction, exhaustive coverage matrices, and targeted "
        "hierarchical synthesis."
    )

    # LLM parameters
    llm_temperature: float = 0.0
    llm_top_p: float = 0.9
    llm_seed: Optional[int] = 42

    # Prompt versions – bump when prompts change to invalidate cache
    prompt_version_planner: str = "v3"
    prompt_version_mapper: str = "v5"
    prompt_version_verifier: str = "v3"
    prompt_version_answer: str = "v5"

    model_config = {
        # The workspace file holds secrets; the project file holds runtime config.
        "env_file": (
            str(WORKSPACE_ROOT / ".env"),
            str(PROJECT_ROOT / ".env"),
        ),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # Derived paths
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        """Create all data directories if they do not exist."""
        for d in [self.uploads_dir, self.parsed_dir, self.runs_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def effective_base_url(self) -> str:
        """Return the effective base URL for OpenAI-compatible endpoints."""
        url = self.llm_base_url or self.ollama_base_url
        if not url:
            return ""
        url = url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url

    def model_policy_warnings(self) -> list[str]:
        """Return model-family violations for the capstone evaluation."""
        warnings: list[str] = []
        for stage, model in {
            "planner": self.planner_model,
            "mapper": self.mapper_model,
            "verifier": self.verifier_model,
            "answer": self.answer_model,
        }.items():
            normalized = model.lower().replace("_", "-")
            accepted_aliases = BROKER_MODEL_IDS | {
                "mistral-small-2506",
                "mistral-small3.2:latest",
                "ministral-8b-2512",
                "ministral-3:latest",
            }
            if normalized not in accepted_aliases:
                warnings.append(
                    f"{stage} model '{model}' is outside the allowed "
                    "course model set"
                )
        return warnings

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/") if v else v


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
