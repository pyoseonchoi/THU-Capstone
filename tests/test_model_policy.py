"""Tests for capstone model-family enforcement."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.exceptions import LLMConfigurationError
from app.pipeline import FullScanPipeline


def test_local_model_produces_warning_without_blocking():
    settings = Settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        planner_model="qwen2.5:7b",
        mapper_model="qwen2.5:7b",
        answer_model="qwen2.5:7b",
        verifier_model="qwen2.5:7b",
        evaluation_mode=False,
    )
    assert len(settings.model_policy_warnings()) == 4


def test_allowed_school_models_pass_policy():
    settings = Settings(
        planner_model="mistral-small3.2:latest",
        mapper_model="ministral-3:latest",
        answer_model="mistral-small-2506",
        verifier_model="mistral-small3.2:latest",
        evaluation_mode=True,
    )
    assert settings.model_policy_warnings() == []


def test_evaluation_mode_rejects_disallowed_models():
    settings = Settings(
        planner_model="qwen2.5:7b",
        mapper_model="qwen2.5:7b",
        answer_model="qwen2.5:7b",
        verifier_model="qwen2.5:7b",
        evaluation_mode=True,
    )
    with pytest.raises(LLMConfigurationError):
        FullScanPipeline(settings)
