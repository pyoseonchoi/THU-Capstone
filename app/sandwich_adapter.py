"""Adapter that runs hdr2025/pipeline_g14.py as the answering engine for
this backend, behind the same /answer-jobs contract the dashboard expects.

pipeline_g14 is a standalone batch script (answers a whole question set in
one call, no per-quote page numbers, its own broker client). This module
wraps it so the FastAPI job endpoints can drive it without needing to
understand its internals.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HDR2025_DIR = REPO_ROOT / "hdr2025"
if str(HDR2025_DIR) not in sys.path:
    sys.path.insert(0, str(HDR2025_DIR))

# WORKERS is read once at import time inside pipeline_g14 (baked into a
# ThreadPoolExecutor(max_workers=WORKERS) constant) -- per the pipeline
# author's own instructions this must be fixed at 2, and it must be set
# before the module is imported or the default of 3 wins instead.
os.environ.setdefault("WORKERS", "2")

# pipeline_ed2.get_client() reads BROKER_API_KEY straight from os.environ,
# but this app's Settings (pydantic-settings) only populates its own
# declared fields from .env -- it never mutates the real process
# environment. Bridge the two explicitly before pipeline_g14 is imported.
# Same story for TEAM (pipeline_g14's own submission["team"] field) vs this
# app's TEAM_NAME setting.
if not os.environ.get("BROKER_API_KEY") or not os.environ.get("TEAM"):
    from app.config import get_settings

    _settings = get_settings()
    if not os.environ.get("BROKER_API_KEY") and _settings.mistral_api_key:
        os.environ["BROKER_API_KEY"] = _settings.mistral_api_key
    if not os.environ.get("TEAM") and _settings.team_name:
        os.environ["TEAM"] = _settings.team_name

import pipeline_g14  # noqa: E402  (path/env must be patched before this import)

_PROGRESS_LINE_RE = re.compile(r"^\s*\[\s*(?:reused|[\d.]+s)\]\s+(\S+)")

# pipeline_g14 never records token usage itself -- it calls the broker
# client directly via pipeline_ed2.get_client() and only ever reads
# response.choices[0].message. We patch the client's create() once (it's a
# cached singleton, so this covers every call site in the file) to also
# tally usage per model, so the dashboard can show real numbers instead of
# a hardcoded 0.
_usage_lock = threading.Lock()
_usage_totals: dict[str, dict[str, int]] = {}
_usage_tracking_installed = False


def _install_usage_tracking() -> None:
    global _usage_tracking_installed
    if _usage_tracking_installed:
        return
    client = pipeline_g14.get_client()
    original_create = client.chat.completions.create

    def tracked_create(*args, **kwargs):
        response = original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            model = kwargs.get("model", "unknown")
            with _usage_lock:
                bucket = _usage_totals.setdefault(model, {"input": 0, "output": 0})
                bucket["input"] += usage.prompt_tokens or 0
                bucket["output"] += usage.completion_tokens or 0
        return response

    client.chat.completions.create = tracked_create
    _usage_tracking_installed = True


def _run_pipeline_g14(document_path: str, questions_path: str, output_path: str) -> dict:
    """Runs on a worker thread via asyncio.to_thread. pipeline_g14's own
    preflight check raises SystemExit (not Exception) on a bad/missing
    broker key -- asyncio's Task machinery deliberately does NOT catch
    SystemExit crossing an await, so left alone it kills the whole server
    process instead of just failing this one job. Convert it here, before
    it crosses back into the event loop."""
    try:
        return pipeline_g14.run(document_path, questions_path, output_path)
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc

ProgressCallback = Callable[[int, int], None]


class UnsupportedDocumentError(Exception):
    pass


def _strip_evidence_text(evidence_block: list[str] | None) -> str:
    if not evidence_block:
        return ""
    text_part = next((e for e in evidence_block if e.startswith("TEXT EVIDENCE:")), "")
    return text_part.removeprefix("TEXT EVIDENCE:").strip()


async def run_sandwich_job(
    document_path: Path,
    questions: list[dict],
    progress_cb: ProgressCallback,
) -> dict:
    """Run pipeline_g14 over the full question set. Returns a dict shaped
    like the dashboard's expected job result (answers/usage/submission)."""
    if document_path.suffix.lower() != ".txt":
        raise UnsupportedDocumentError(
            "The Sandwich model only supports .txt documents "
            "(it reads the file as plain text; a .pdf would be garbled)."
        )

    await asyncio.to_thread(_install_usage_tracking)
    with _usage_lock:
        usage_before = {model: dict(counts) for model, counts in _usage_totals.items()}

    total = len(questions)
    processed = 0
    loop = asyncio.get_running_loop()

    def on_log_line(message: str) -> None:
        nonlocal processed
        match = _PROGRESS_LINE_RE.match(message)
        if match:
            processed += 1
            loop.call_soon_threadsafe(progress_cb, processed, total)
        print(message, flush=True)

    with tempfile.TemporaryDirectory(prefix="sandwich-job-") as tmpdir:
        questions_path = Path(tmpdir) / "questions.yaml"
        output_path = Path(tmpdir) / "submission.json"
        questions_path.write_text(
            yaml.safe_dump(questions, allow_unicode=True), encoding="utf-8"
        )

        original_log = pipeline_g14._log
        pipeline_g14._log = on_log_line
        try:
            submission = await asyncio.to_thread(
                _run_pipeline_g14, str(document_path), str(questions_path), str(output_path)
            )
        finally:
            pipeline_g14._log = original_log

    answers = []
    for entry in submission.get("answers", []):
        evidence_text = _strip_evidence_text(entry.get("evidence"))
        answers.append({
            "question_id": entry["id"],
            "final_answer": entry.get("answer", ""),
            "warnings": [entry["error"]] if entry.get("error") else [],
            "evidence_quotes": (
                [{"quote": evidence_text, "page": None}] if evidence_text else []
            ),
            "source_pages": [],
        })

    with _usage_lock:
        usage_after = {model: dict(counts) for model, counts in _usage_totals.items()}

    by_model: dict[str, dict[str, int]] = {}
    for model, after in usage_after.items():
        before = usage_before.get(model, {"input": 0, "output": 0})
        by_model[model] = {
            "input_tokens": after["input"] - before["input"],
            "output_tokens": after["output"] - before["output"],
        }
    total_input = sum(m["input_tokens"] for m in by_model.values())
    total_output = sum(m["output_tokens"] for m in by_model.values())

    return {
        "answers": answers,
        "usage": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_model": by_model,
        },
        "submission": submission,
    }
