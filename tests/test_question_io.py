"""Tests for official question-set input parsing."""

from __future__ import annotations

import pytest

from app.question_io import parse_questions_json, parse_questions_text


def test_text_preserves_explicit_question_ids():
    questions = parse_questions_text("q01: First?\nq02\tSecond?")
    assert [question.question_id for question in questions] == ["q01", "q02"]


def test_plain_lines_receive_stable_ids():
    questions = parse_questions_text("First?\nSecond?")
    assert [question.question_id for question in questions] == ["q01", "q02"]


def test_json_accepts_grader_shape():
    questions = parse_questions_json(
        '{"questions":[{"id":"q01","question":"First?"}]}'
    )
    assert questions[0].question_id == "q01"


def test_json_preserves_optional_category_hint():
    questions = parse_questions_json(
        '[{"id":"d10","category":"contradiction","question":"Conflict?"}]'
    )
    assert questions[0].category == "contradiction"


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_questions_text("q01: First?\nq01: Second?")
