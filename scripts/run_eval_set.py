"""Run one evaluation set and save submission plus token/time summaries.

Expected dataset shape:

    data/eval_sets/<name>/
      document.txt or document.pdf
      questions.json

This script does not score answers. Scoring needs a frozen answer_key.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.logging_config import setup_logging
from app.pipeline import FullScanPipeline
from app.question_io import parse_questions_json


class OfflineDeterministicClient(BaseLLMClient):
    """Fail loudly if an evaluation expected to be deterministic reaches the LLM."""

    async def chat(self, messages, **kwargs) -> LLMResponse:
        del messages
        stage = kwargs.get("stage", "unknown")
        question_ids = kwargs.get("question_ids") or []
        raise RuntimeError(
            "Offline deterministic evaluation reached an LLM stage "
            f"({stage}) for questions {question_ids}."
        )

    async def close(self) -> None:
        return None


def _usage_by_stage(records: list) -> dict[str, dict]:
    stages: dict[str, dict] = {}
    for record in records:
        stage = record.stage or "unknown"
        if stage not in stages:
            stages[stage] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "failures": 0,
            }
        bucket = stages[stage]
        bucket["calls"] += 1
        bucket["input_tokens"] += record.input_tokens or 0
        bucket["output_tokens"] += record.output_tokens or 0
        bucket["latency_ms"] += round(record.latency_ms, 1)
        if not record.success:
            bucket["failures"] += 1
    return stages


def _append_summary_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


async def _run(args: argparse.Namespace) -> dict:
    setup_logging()
    settings = get_settings()
    settings.finalize_deterministic_with_llm = args.llm_finalize_deterministic
    settings.ensure_dirs()

    questions = parse_questions_json(args.questions.read_bytes())
    dataset_name = args.name or args.document.parent.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / dataset_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_client = OfflineDeterministicClient() if args.offline_deterministic else None
    pipeline = FullScanPipeline(settings, llm_client=llm_client)
    started = time.perf_counter()
    try:
        process_started = time.perf_counter()
        metadata, chunks = await pipeline.process_document(args.document)
        process_seconds = time.perf_counter() - process_started

        answer_started = time.perf_counter()
        run_result = await pipeline.answer_questions(metadata.document_id, questions)
        answer_seconds = time.perf_counter() - answer_started
    finally:
        await pipeline.close()

    wall_seconds = time.perf_counter() - started
    submission_source = Path(run_result.submission_path)
    submission_target = run_dir / "submission.json"
    if submission_source.exists():
        shutil.copy2(submission_source, submission_target)

    run_json = run_dir / "run_result.json"
    run_json.write_text(run_result.model_dump_json(indent=2), encoding="utf-8")

    total_tokens = run_result.total_input_tokens + run_result.total_output_tokens
    failed_calls = sum(1 for record in run_result.usage if not record.success)
    summary = {
        "dataset": dataset_name,
        "document": str(args.document),
        "questions": str(args.questions),
        "output_dir": str(run_dir),
        "submission": str(submission_target),
        "question_count": len(questions),
        "document_id": metadata.document_id,
        "pages": metadata.page_count,
        "chunks": len(chunks),
        "llm_calls": len(run_result.usage),
        "failed_llm_calls": failed_calls,
        "input_tokens": run_result.total_input_tokens,
        "output_tokens": run_result.total_output_tokens,
        "total_tokens": total_tokens,
        "process_seconds": round(process_seconds, 2),
        "answer_seconds": round(answer_seconds, 2),
        "wall_seconds": round(wall_seconds, 2),
        "pipeline_latency_seconds": round(run_result.total_latency_ms / 1000, 2),
        "usage_by_stage": _usage_by_stage(run_result.usage),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_row = {
        "dataset": dataset_name,
        "run_dir": str(run_dir),
        "questions": len(questions),
        "pages": metadata.page_count,
        "chunks": len(chunks),
        "llm_calls": len(run_result.usage),
        "input_tokens": run_result.total_input_tokens,
        "output_tokens": run_result.total_output_tokens,
        "total_tokens": total_tokens,
        "wall_seconds": round(wall_seconds, 2),
        "score": "",
        "notes": "score requires answer_key.yaml",
    }
    _append_summary_csv(args.output_root / "runs_summary.csv", csv_row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one FULLSCAN-QA evaluation set.")
    parser.add_argument("document", type=Path, help="Path to document.txt or document.pdf")
    parser.add_argument("questions", type=Path, help="Path to questions.json")
    parser.add_argument("--name", default="", help="Dataset name for output folders")
    parser.add_argument(
        "--offline-deterministic",
        action="store_true",
        help="Do not call any LLM; fail if a question cannot be answered deterministically.",
    )
    parser.add_argument(
        "--llm-finalize-deterministic",
        action="store_true",
        help="Call the answer LLM once per deterministic answer for token/time measurement.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/eval_runs"),
        help="Where run artifacts and summary CSV should be written",
    )
    args = parser.parse_args()
    if args.offline_deterministic and args.llm_finalize_deterministic:
        parser.error("--offline-deterministic cannot be combined with --llm-finalize-deterministic")

    summary = asyncio.run(_run(args))
    print("\nEvaluation run complete")
    print(f"  output:        {summary['output_dir']}")
    print(f"  submission:    {summary['submission']}")
    print(f"  questions:     {summary['question_count']}")
    print(f"  chunks:        {summary['chunks']}")
    print(f"  LLM calls:     {summary['llm_calls']}")
    print(f"  input tokens:  {summary['input_tokens']:,}")
    print(f"  output tokens: {summary['output_tokens']:,}")
    print(f"  total tokens:  {summary['total_tokens']:,}")
    print(f"  wall seconds:  {summary['wall_seconds']}")


if __name__ == "__main__":
    main()
