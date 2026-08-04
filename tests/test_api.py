"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import app
from app.schemas import DocumentChunk, DocumentMetadata, PipelineAnswer, PipelineRun, QuestionRequest


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "fullscan-qa"
        assert "version" in data


class TestCORS:
    def test_cors_allows_cross_origin_requests(self, client):
        response = client.get("/health", headers={"Origin": "http://127.0.0.1:5500"})
        assert response.headers.get("access-control-allow-origin") == "*"


class TestAnswerJobs:
    def test_invalid_pipeline_mode_is_rejected(self, client):
        response = client.post(
            "/answer-jobs",
            json={
                "document_id": "missing",
                "questions": [{"question_id": "q1", "question": "test"}],
                "pipeline_mode": "INVALID",
            },
        )
        assert response.status_code == 400

    def test_unknown_job_returns_404(self, client):
        response = client.get("/answer-jobs/does-not-exist")
        assert response.status_code == 404


class TestAnswerJobProgressQueue:
    async def test_progress_events_are_queued_in_order(self, monkeypatch):
        class FakeJobPipeline:
            async def answer_questions(self, document_id, questions, mode=None, progress_callback=None):
                if progress_callback:
                    progress_callback("mapping", 1, 2, 0)
                    progress_callback("answering", 2, 2, 0)
                return PipelineRun(
                    document_id=document_id,
                    answers=[PipelineAnswer(question_id="q1", final_answer="answer")],
                )

            async def close(self):
                return None

        async def fake_llm_health(settings):
            return {"available": True, "message": "ok"}

        monkeypatch.setattr(api_main, "_get_pipeline", lambda: FakeJobPipeline())
        monkeypatch.setattr(api_main, "check_llm_health", fake_llm_health)

        job_id = "job-progress-1"
        api_main._answer_jobs[job_id] = {
            "job_id": job_id, "status": "running", "stage": "compiling",
            "processed": 0, "total": 0, "failed": 0,
        }
        api_main._answer_job_queues[job_id] = asyncio.Queue()

        req = api_main.AnswerRequest(
            document_id="doc-1",
            questions=[QuestionRequest(question_id="q1", question="test?")],
        )
        await api_main._execute_answer_job(job_id, req)

        queue = api_main._answer_job_queues[job_id]
        events = []
        while True:
            item = queue.get_nowait()
            if item is None:
                break
            events.append(item)

        assert events == [
            {"stage": "mapping", "processed": 1, "total": 2, "failed": 0},
            {"stage": "answering", "processed": 2, "total": 2, "failed": 0},
        ]
        assert api_main._answer_jobs[job_id]["status"] == "completed"


class TestAnswerJobStream:
    async def test_stream_emits_queued_events_then_done(self):
        job_id = "job-stream-1"
        api_main._answer_jobs[job_id] = {"job_id": job_id, "status": "running"}
        queue = asyncio.Queue()
        queue.put_nowait({"stage": "mapping", "processed": 1, "total": 2, "failed": 0})
        queue.put_nowait(None)
        api_main._answer_job_queues[job_id] = queue

        response = await api_main.stream_answer_job(job_id)
        chunks = [chunk async for chunk in response.body_iterator]
        body = "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

        assert '"stage": "mapping"' in body
        assert body.strip().endswith("event: done\ndata: {}")

    async def test_stream_of_unknown_job_returns_404(self):
        with pytest.raises(HTTPException) as exc_info:
            await api_main.stream_answer_job("does-not-exist")
        assert exc_info.value.status_code == 404

    async def test_stream_of_finished_job_replays_final_state_immediately(self):
        job_id = "job-stream-2"
        api_main._answer_jobs[job_id] = {
            "job_id": job_id, "status": "completed", "stage": "answering",
        }
        api_main._answer_job_queues.pop(job_id, None)

        response = await api_main.stream_answer_job(job_id)
        chunks = [chunk async for chunk in response.body_iterator]
        body = "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

        assert '"status": "completed"' in body
        assert "event: done" in body


class TestDocumentUpload:
    def test_txt_upload_is_accepted(self, client, monkeypatch, tmp_path):
        class FakeStore:
            def save_upload(self, filename, content):
                path = tmp_path / filename
                path.write_bytes(content)
                return path

        class FakePipeline:
            async def process_document(self, path):
                assert path.suffix == ".txt"
                metadata = DocumentMetadata(
                    document_id="text-doc",
                    filename=path.name,
                    page_count=1,
                    chunk_count=1,
                )
                chunk = DocumentChunk(
                    document_id="text-doc",
                    chunk_id="text-chunk",
                    chunk_index=0,
                    text="plain text",
                )
                return metadata, [chunk]

            async def close(self):
                return None

        monkeypatch.setattr(api_main, "_get_store", lambda: FakeStore())
        monkeypatch.setattr(api_main, "_get_pipeline", lambda: FakePipeline())

        response = client.post(
            "/documents",
            files={"file": ("notes.txt", b"plain text", "text/plain")},
        )

        assert response.status_code == 200
        assert response.json()["document_id"] == "text-doc"
