"""Competition submission JSON construction and incremental persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import PipelineAnswer, PipelineRun, QuestionRequest


def answer_to_submission(answer: PipelineAnswer) -> dict[str, Any]:
    """Convert one internal answer to the grader's minimal interface."""
    answer_text = answer.final_answer.strip()
    if answer_text.startswith("Error:") and answer.operation_result:
        fallback = answer.operation_result.result_value
        if isinstance(fallback, str) and fallback.strip():
            answer_text = fallback.strip()
    failed = not answer_text or answer_text.startswith("Error:")
    entry: dict[str, Any] = {
        "id": answer.question_id,
        "answer": "" if failed else answer_text,
    }
    evidence: list[str] = []
    if answer.answer_with_evidence:
        evidence.append(answer.answer_with_evidence)
    if answer.operation_result and answer.operation_result.source_pages:
        pages = ", ".join(str(page) for page in answer.operation_result.source_pages)
        evidence.append(f"Source pages/segments: {pages}")
    if evidence:
        entry["evidence"] = evidence
    if failed:
        entry["error"] = "; ".join(answer.warnings) or answer.final_answer
    return entry


def build_submission(
    run: PipelineRun,
    *,
    team: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build a complete grader-compatible submission payload."""
    payload: dict[str, Any] = {
        "answers": [answer_to_submission(answer) for answer in run.answers],
    }
    if team.strip():
        payload["team"] = team.strip()
    if notes.strip():
        payload["notes"] = notes.strip()
    return payload


class IncrementalSubmissionWriter:
    """Persist a valid partial submission after every completed question."""

    def __init__(self, path: Path, *, team: str = "", notes: str = ""):
        self.path = path
        self.team = team.strip()
        self.notes = notes.strip()
        self._answers: dict[str, dict[str, Any]] = {}

    def initialize(self, questions: list[QuestionRequest]) -> Path:
        self._answers = {
            question.question_id: {
                "id": question.question_id,
                "answer": "",
                "error": "Pending",
            }
            for question in questions
        }
        return self._write()

    def update(self, answer: PipelineAnswer) -> Path:
        self._answers[answer.question_id] = answer_to_submission(answer)
        return self._write()

    def finalize(self, run: PipelineRun) -> Path:
        for answer in run.answers:
            self._answers[answer.question_id] = answer_to_submission(answer)
        return self._write()

    def _write(self) -> Path:
        payload: dict[str, Any] = {
            "answers": list(self._answers.values()),
        }
        if self.team:
            payload["team"] = self.team
        if self.notes:
            payload["notes"] = self.notes

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return self.path
