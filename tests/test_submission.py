"""Tests for grader-compatible incremental submissions."""

from __future__ import annotations

import json

from app.schemas import PipelineAnswer, PipelineRun, QuestionRequest
from app.submission import IncrementalSubmissionWriter, build_submission


def test_submission_has_only_required_answer_interface():
    run = PipelineRun(
        document_id="doc",
        answers=[PipelineAnswer(question_id="q01", final_answer="Seven")],
    )
    payload = build_submission(run, team="team-a", notes="method")
    assert payload["team"] == "team-a"
    assert payload["answers"] == [{"id": "q01", "answer": "Seven"}]


def test_incremental_writer_keeps_pending_questions(tmp_path):
    path = tmp_path / "submission.json"
    writer = IncrementalSubmissionWriter(path, notes="method")
    writer.initialize([
        QuestionRequest(question_id="q01", question="One?"),
        QuestionRequest(question_id="q02", question="Two?"),
    ])
    writer.update(PipelineAnswer(question_id="q01", final_answer="Answer one"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {answer["id"]: answer for answer in payload["answers"]}
    assert by_id["q01"]["answer"] == "Answer one"
    assert by_id["q02"]["error"] == "Pending"
