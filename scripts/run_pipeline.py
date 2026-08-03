"""CLI pipeline runner for FULLSCAN-QA."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import PipelineMode, get_settings
from app.logging_config import setup_logging
from app.pipeline import FullScanPipeline
from app.schemas import QuestionRequest


def main():
    parser = argparse.ArgumentParser(description="FULLSCAN-QA Pipeline Runner")
    parser.add_argument("pdf", type=Path, help="Path to PDF document")
    parser.add_argument(
        "-q", "--questions", nargs="+", required=True,
        help="One or more questions to answer",
    )
    parser.add_argument(
        "-m", "--mode", default="FULLSCAN_OPERATOR",
        choices=["FULLSCAN_OPERATOR", "DIRECT_CONTEXT", "SUMMARY_MAP_REDUCE", "REFINE"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON file path",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    questions = [
        QuestionRequest(question_id=f"q{i+1}", question=q)
        for i, q in enumerate(args.questions)
    ]
    mode = PipelineMode(args.mode)

    async def run():
        pipeline = FullScanPipeline(settings)
        try:
            metadata, chunks = await pipeline.process_document(args.pdf)
            print(
                f"Document: {metadata.filename}, {metadata.page_count} pages, "
                f"{len(chunks)} chunks"
            )

            run_result = await pipeline.answer_questions(
                metadata.document_id, questions, mode=mode
            )

            for ans in run_result.answers:
                print(f"\n{'='*60}")
                print(f"Q: {ans.question_id}")
                print(f"Operator: {ans.operator}")
                print(f"Answer: {ans.final_answer}")
                if ans.coverage:
                    print(f"Coverage: {ans.coverage.coverage_percentage}%")
                if ans.warnings:
                    print(f"Warnings: {ans.warnings}")

            if args.output:
                args.output.write_text(
                    run_result.model_dump_json(indent=2), encoding="utf-8"
                )
                print(f"\nResults saved to {args.output}")

        finally:
            await pipeline.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
