"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import app
from app.schemas import DocumentChunk, DocumentMetadata


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
