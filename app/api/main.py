"""FastAPI endpoints for FULLSCAN-QA."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import PipelineMode, get_settings
from app.llm.broker import fetch_broker_usage
from app.llm.health import check_llm_health
from app.logging_config import setup_logging
from app.pipeline import FullScanPipeline
from app.schemas import PipelineRun, QuestionRequest
from app.storage.run_store import RunStore
from app.submission import build_submission

setup_logging()

app = FastAPI(
    title="FULLSCAN-QA",
    description=(
        "Adaptive exhaustive large-document QA without RAG. "
        "This system does not use embeddings, vector search, semantic retrieval, "
        "reranking, or top-k context selection."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".txt")

def _get_pipeline() -> FullScanPipeline:
    settings = get_settings()
    settings.ensure_dirs()
    return FullScanPipeline(settings)


def _get_store() -> RunStore:
    settings = get_settings()
    settings.ensure_dirs()
    return RunStore(settings)


# ---------- Health ----------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "fullscan-qa", "version": "3.0.0"}


@app.get("/health/llm")
async def llm_health():
    """Check that the configured LLM endpoint and models are ready."""
    return await check_llm_health(get_settings())


@app.get("/usage")
async def broker_usage():
    """Return the course broker's metered usage for this group."""
    try:
        return await fetch_broker_usage(get_settings())
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


# ---------- Documents ----------

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    warnings: list[str] = []


@app.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and parse a PDF or plain-text document."""
    if not file.filename or not file.filename.lower().endswith(
        SUPPORTED_DOCUMENT_EXTENSIONS
    ):
        raise HTTPException(400, "Only PDF and TXT files are supported.")

    pipeline = _get_pipeline()
    store = _get_store()

    # Save upload to temp location
    content = await file.read()
    upload_path = store.save_upload(file.filename, content)

    try:
        metadata, chunks = await pipeline.process_document(upload_path)
    except Exception as exc:
        setup_logging()
        from app.logging_config import get_logger
        get_logger("api").exception("Document parsing failed: %s", exc)
        raise HTTPException(500, f"Document parsing failed: {exc}") from exc
    finally:
        await pipeline.close()

    return DocumentResponse(
        document_id=metadata.document_id,
        filename=metadata.filename,
        page_count=metadata.page_count,
        chunk_count=metadata.chunk_count,
    )


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Return document metadata and coverage manifest."""
    store = _get_store()
    meta = store.load_document_metadata(document_id)
    if not meta:
        raise HTTPException(404, "Document not found")
    return meta.model_dump(mode="json")


# ---------- Answer ----------

class AnswerRequest(BaseModel):
    document_id: str
    questions: list[QuestionRequest]
    include_diagnostics: bool = True
    pipeline_mode: str = "ADAPTIVE_HIERARCHICAL"


def _serialize_run(run: PipelineRun) -> dict:
    settings = get_settings()
    return {
        "run_id": run.run_id,
        "document_id": run.document_id,
        "answers": [
            {
                "question_id": answer.question_id,
                "final_answer": answer.final_answer,
                "answer_with_evidence": answer.answer_with_evidence,
                "operator": answer.operator.value if answer.operator else None,
                "coverage": (
                    answer.coverage.model_dump(mode="json")
                    if answer.coverage
                    else {}
                ),
                "verification": [
                    item.model_dump(mode="json")
                    for item in answer.verification
                ],
                "warnings": answer.warnings,
            }
            for answer in run.answers
        ],
        "usage": {
            "total_input_tokens": run.total_input_tokens,
            "total_output_tokens": run.total_output_tokens,
        },
        "timing": {"total_latency_ms": round(run.total_latency_ms, 1)},
        "diagnostics": {
            "pipeline_mode": run.pipeline_mode,
            "warnings": run.warnings,
        },
        "submission": build_submission(
            run,
            team=settings.team_name,
            notes=settings.submission_notes,
        ),
    }


@app.post("/answer")
async def answer_questions(req: AnswerRequest):
    """Answer questions about a parsed document."""
    pipeline = _get_pipeline()

    try:
        mode = PipelineMode(req.pipeline_mode)
    except ValueError:
        raise HTTPException(400, f"Invalid mode: {req.pipeline_mode}")

    try:
        run = await pipeline.answer_questions(
            req.document_id, req.questions, mode=mode
        )
        result = _serialize_run(run)
        if not req.include_diagnostics:
            result.pop("diagnostics", None)
        return result
    except Exception as exc:
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc
    finally:
        await pipeline.close()


_answer_jobs: dict[str, dict] = {}
_answer_job_queues: dict[str, asyncio.Queue] = {}


async def _execute_answer_job(job_id: str, req: AnswerRequest) -> None:
    pipeline: FullScanPipeline | None = None
    queue = _answer_job_queues.get(job_id)
    try:
        mode = PipelineMode(req.pipeline_mode)
        llm_status = await check_llm_health(get_settings())
        if not llm_status["available"]:
            raise RuntimeError(llm_status["message"])
        pipeline = _get_pipeline()

        def progress(stage: str, processed: int, total: int, failed: int) -> None:
            event = {
                "stage": stage,
                "processed": processed,
                "total": total,
                "failed": failed,
            }
            _answer_jobs[job_id].update(event)
            if queue is not None:
                queue.put_nowait(event)

        run = await pipeline.answer_questions(
            req.document_id,
            req.questions,
            mode=mode,
            progress_callback=progress,
        )
        result = _serialize_run(run)
        if not req.include_diagnostics:
            result.pop("diagnostics", None)
        _answer_jobs[job_id].update({
            "status": "completed",
            "result": result,
        })
    except Exception as exc:
        _answer_jobs[job_id].update({
            "status": "failed",
            "error": str(exc),
        })
    finally:
        if pipeline is not None:
            await pipeline.close()
        if queue is not None:
            queue.put_nowait(None)


@app.post("/answer-jobs", status_code=202)
async def create_answer_job(req: AnswerRequest):
    """Start a long-running answer job and return immediately."""
    try:
        PipelineMode(req.pipeline_mode)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid mode: {req.pipeline_mode}") from exc

    job_id = uuid.uuid4().hex[:16]
    _answer_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "stage": "compiling",
        "processed": 0,
        "total": 0,
        "failed": 0,
    }
    _answer_job_queues[job_id] = asyncio.Queue()
    asyncio.create_task(_execute_answer_job(job_id, req))
    return _answer_jobs[job_id]


@app.get("/answer-jobs/{job_id}")
async def get_answer_job(job_id: str):
    """Return progress or the final result for an answer job."""
    job = _answer_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Answer job not found")
    return job


# ---------- Direct answer (multipart) ----------

@app.post("/answer-direct")
async def answer_direct(
    file: UploadFile = File(...),
    questions_json: str = Form(...),
):
    """Upload a PDF or TXT document and answer questions in one request."""
    import json

    pipeline = _get_pipeline()
    store = _get_store()

    if not file.filename or not file.filename.lower().endswith(
        SUPPORTED_DOCUMENT_EXTENSIONS
    ):
        raise HTTPException(400, "Only PDF and TXT files are supported.")

    try:
        questions_data = json.loads(questions_json)
        questions = [QuestionRequest(**q) for q in questions_data]
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid questions JSON: {exc}")

    content = await file.read()
    upload_path = store.save_upload(file.filename, content)
    try:
        metadata, chunks = await pipeline.process_document(upload_path)
        run = await pipeline.answer_questions(metadata.document_id, questions)
        return {
            "run_id": run.run_id,
            "document_id": metadata.document_id,
            "answers": [
                {
                    "question_id": answer.question_id,
                    "final_answer": answer.final_answer,
                    "operator": (
                        answer.operator.value if answer.operator else None
                    ),
                    "warnings": answer.warnings,
                }
                for answer in run.answers
            ],
        }
    finally:
        await pipeline.close()


# ---------- Runs ----------

@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Retrieve a completed pipeline run."""
    store = _get_store()
    run = store.load_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run.model_dump(mode="json")


@app.get("/runs/{run_id}/submission")
async def get_submission(run_id: str):
    """Return the grader-compatible JSON for a completed or partial run."""
    settings = get_settings()
    path = settings.runs_dir / f"{run_id}_submission.json"
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    store = _get_store()
    run = store.load_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return build_submission(
        run,
        team=settings.team_name,
        notes=settings.submission_notes,
    )
