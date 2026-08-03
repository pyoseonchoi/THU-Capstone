"""Token and performance usage tracker."""

from __future__ import annotations

from app.logging_config import get_logger
from app.schemas import UsageRecord

logger = get_logger("llm.usage_tracker")


class UsageTracker:
    """Accumulates usage records and provides aggregation."""

    def __init__(self):
        self._records: list[UsageRecord] = []

    def record(self, usage: UsageRecord) -> None:
        """Add a usage record."""
        self._records.append(usage)

    @property
    def records(self) -> list[UsageRecord]:
        return list(self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens or 0 for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens or 0 for r in self._records)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self._records)

    @property
    def total_calls(self) -> int:
        return len(self._records)

    @property
    def failed_calls(self) -> int:
        return sum(1 for r in self._records if not r.success)

    def aggregate_by_stage(self) -> dict[str, dict]:
        """Aggregate usage by pipeline stage."""
        stages: dict[str, dict] = {}
        for r in self._records:
            if r.stage not in stages:
                stages[r.stage] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.0,
                    "failures": 0,
                }
            s = stages[r.stage]
            s["calls"] += 1
            s["input_tokens"] += r.input_tokens or 0
            s["output_tokens"] += r.output_tokens or 0
            s["latency_ms"] += r.latency_ms
            if not r.success:
                s["failures"] += 1
        return stages

    def aggregate_by_model(self) -> dict[str, dict]:
        """Aggregate usage by model."""
        models: dict[str, dict] = {}
        for r in self._records:
            if r.model not in models:
                models[r.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.0,
                }
            m = models[r.model]
            m["calls"] += 1
            m["input_tokens"] += r.input_tokens or 0
            m["output_tokens"] += r.output_tokens or 0
            m["latency_ms"] += r.latency_ms
        return models

    def summary(self) -> dict:
        """Return a summary dict."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "failed_calls": self.failed_calls,
            "by_stage": self.aggregate_by_stage(),
            "by_model": self.aggregate_by_model(),
        }
