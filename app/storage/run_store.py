"""Filesystem-based run storage.

Persists uploaded documents, parsed pages, chunks, plans,
map outputs, evidence ledgers, and final answers as JSON files.
Atomic writes to reduce corruption risk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.config import Settings
from app.logging_config import get_logger
from app.schemas import (
    DocumentChunk,
    DocumentMetadata,
    PipelineRun,
)
from app.v3.models import CompiledDocument

logger = get_logger("storage.run_store")


def _atomic_write(path: Path, data: str) -> None:
    """Write data atomically using a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp"
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            f.write(data)
        shutil.move(tmp_path, str(path))
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise


class RunStore:
    """Manages filesystem persistence for pipeline runs."""

    def __init__(self, settings: Settings):
        self._settings = settings
        settings.ensure_dirs()

    def save_document_metadata(self, meta: DocumentMetadata) -> Path:
        """Save document metadata."""
        path = self._settings.parsed_dir / f"{meta.document_id}_meta.json"
        _atomic_write(path, meta.model_dump_json(indent=2))
        return path

    def save_chunks(
        self, document_id: str, chunks: list[DocumentChunk]
    ) -> Path:
        """Save document chunks."""
        path = self._settings.parsed_dir / f"{document_id}_chunks.json"
        data = json.dumps(
            [c.model_dump(mode="json") for c in chunks], indent=2
        )
        _atomic_write(path, data)
        return path

    def load_chunks(self, document_id: str) -> list[DocumentChunk]:
        """Load chunks for a document."""
        path = self._settings.parsed_dir / f"{document_id}_chunks.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [DocumentChunk(**c) for c in data]

    def load_document_metadata(self, document_id: str) -> DocumentMetadata | None:
        """Load document metadata."""
        path = self._settings.parsed_dir / f"{document_id}_meta.json"
        if not path.exists():
            return None
        return DocumentMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    def save_compiled_document(self, document: CompiledDocument) -> Path:
        """Persist the question-independent V3 document representation."""
        path = self._settings.parsed_dir / f"{document.document_id}_compiled_v3.json"
        _atomic_write(path, document.model_dump_json(indent=2))
        return path

    def load_compiled_document(self, document_id: str) -> CompiledDocument | None:
        """Load a previously compiled V3 document."""
        path = self._settings.parsed_dir / f"{document_id}_compiled_v3.json"
        if not path.exists():
            return None
        return CompiledDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def load_quality_records(self, document_id: str) -> list:
        """Load page quality records from parsed output."""
        path = self._settings.parsed_dir / f"{document_id}_parsed.json"
        if not path.exists():
            return []
        try:
            from app.parsing.pdf_parser import load_parsed_output
            _, pages = load_parsed_output(path)
            return [p.quality for p in pages if p.quality]
        except Exception as exc:
            logger.warning("Could not load quality records for %s: %s", document_id, exc)
            return []

    def save_run(self, run: PipelineRun) -> Path:
        """Save a complete pipeline run."""
        path = self._settings.runs_dir / f"{run.run_id}.json"
        _atomic_write(path, run.model_dump_json(indent=2))
        logger.info("Saved run %s to %s", run.run_id, path)
        return path

    def load_run(self, run_id: str) -> PipelineRun | None:
        """Load a pipeline run."""
        path = self._settings.runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return PipelineRun.model_validate_json(path.read_text(encoding="utf-8"))

    def save_artifact(
        self, run_id: str, name: str, data: Any
    ) -> Path:
        """Save an arbitrary artifact as JSON."""
        run_dir = self._settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{name}.json"
        _atomic_write(path, json.dumps(data, indent=2, default=str))
        return path

    def save_upload(self, filename: str, content: bytes) -> Path:
        """Save an uploaded file."""
        path = self._settings.uploads_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path
