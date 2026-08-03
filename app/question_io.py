"""Question-set parsing while preserving official grader IDs."""

from __future__ import annotations

import json
import re
from typing import Any

from app.schemas import QuestionRequest


def _validate_unique(questions: list[QuestionRequest]) -> list[QuestionRequest]:
    ids = [question.question_id for question in questions]
    duplicates = sorted({question_id for question_id in ids if ids.count(question_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {', '.join(duplicates)}")
    return questions


def parse_questions_text(text: str) -> list[QuestionRequest]:
    """Parse lines, accepting optional `q01: question` or tab-separated IDs."""
    questions: list[QuestionRequest] = []
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(q[\w.-]+)\s*(?:\t|:|\|)\s*(.+)$", line, re.IGNORECASE)
        if match:
            question_id, question = match.group(1), match.group(2)
        else:
            question_id, question = f"q{index:02d}", line
        questions.append(QuestionRequest(
            question_id=question_id,
            question=question,
        ))
    return _validate_unique(questions)


def parse_questions_json(raw: str | bytes) -> list[QuestionRequest]:
    """Parse a list or `{questions: [...]}` question-set payload."""
    data: Any = json.loads(raw)
    items = data.get("questions", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Question JSON must be a list or contain a questions list")

    questions: list[QuestionRequest] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every question must be a JSON object")
        question_id = item.get("id", item.get("question_id", ""))
        question = item.get("question", item.get("text", ""))
        if not question_id or not question:
            raise ValueError("Every question needs an id and question text")
        questions.append(QuestionRequest(
            question_id=str(question_id),
            question=str(question),
        ))
    return _validate_unique(questions)
