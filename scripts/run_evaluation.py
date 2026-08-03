"""Evaluation harness for FULLSCAN-QA.

Input: JSONL file with questions and reference answers.
Output: CSV and JSON with deterministic metrics.
Does NOT use an LLM for evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import PipelineMode, get_settings
from app.logging_config import setup_logging
from app.pipeline import FullScanPipeline
from app.schemas import QuestionRequest


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:!?]", "", text)
    return text


def compute_metrics(
    predicted: str,
    reference: str,
    key_points: list[str],
    question_type: str,
) -> dict:
    """Compute deterministic evaluation metrics."""
    pred_norm = normalize_for_comparison(predicted)
    ref_norm = normalize_for_comparison(reference)

    metrics: dict = {
        "exact_match": int(pred_norm == ref_norm),
        "normalized_match": int(pred_norm == ref_norm),
    }

    # Key-point coverage
    if key_points:
        matched = sum(
            1 for kp in key_points
            if normalize_for_comparison(kp) in pred_norm
        )
        metrics["key_point_coverage"] = matched / len(key_points)
        metrics["key_points_matched"] = matched
        metrics["key_points_total"] = len(key_points)
    else:
        metrics["key_point_coverage"] = None

    # Type-specific metrics
    if question_type in ("COUNT", "FILTER_COUNT_LIST"):
        # Extract numbers
        import re
        pred_nums = re.findall(r"\d+", predicted)
        ref_nums = re.findall(r"\d+", reference)
        if pred_nums and ref_nums:
            metrics["count_correct"] = int(pred_nums[0] == ref_nums[0])
        else:
            metrics["count_correct"] = 0

    if question_type in ("FILTER_LIST", "FILTER_COUNT_LIST"):
        # List precision/recall
        pred_items = set(normalize_for_comparison(predicted).split(","))
        ref_items = set(normalize_for_comparison(reference).split(","))
        if ref_items:
            tp = len(pred_items & ref_items)
            metrics["list_precision"] = tp / len(pred_items) if pred_items else 0
            metrics["list_recall"] = tp / len(ref_items) if ref_items else 0
        else:
            metrics["list_precision"] = None
            metrics["list_recall"] = None

    if question_type in ("ARGMAX", "ARGMIN", "SUM", "AVERAGE", "DIFFERENCE", "PERCENT_CHANGE"):
        import re
        pred_nums = re.findall(r"[\d.,]+", predicted)
        ref_nums = re.findall(r"[\d.,]+", reference)
        if pred_nums and ref_nums:
            try:
                p = float(pred_nums[0].replace(",", ""))
                r = float(ref_nums[0].replace(",", ""))
                metrics["numeric_correct"] = int(abs(p - r) < 0.01 * abs(r) + 0.001)
            except ValueError:
                metrics["numeric_correct"] = 0

    if question_type == "ABSENCE":
        metrics["absence_correct"] = int(
            normalize_for_comparison(reference) in pred_norm
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="FULLSCAN-QA Evaluation")
    parser.add_argument("document", type=Path, help="Path to the PDF or TXT document")
    parser.add_argument("eval_file", type=Path, help="JSONL evaluation file")
    parser.add_argument(
        "-m", "--mode", default="ADAPTIVE_HIERARCHICAL",
        choices=["ADAPTIVE_HIERARCHICAL"],
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    # Load evaluation data
    eval_items: list[dict] = []
    with open(args.eval_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))

    questions = [
        QuestionRequest(question_id=item["question_id"], question=item["question"])
        for item in eval_items
    ]
    mode = PipelineMode(args.mode)

    async def run():
        pipeline = FullScanPipeline(settings)
        try:
            metadata, chunks = await pipeline.process_document(args.document)
            run_result = await pipeline.answer_questions(
                metadata.document_id, questions, mode=mode
            )
            return run_result
        finally:
            await pipeline.close()

    run_result = asyncio.run(run())

    # Build results
    results: list[dict] = []
    for item in eval_items:
        qid = item["question_id"]
        answer_obj = next(
            (a for a in run_result.answers if a.question_id == qid), None
        )
        predicted = answer_obj.final_answer if answer_obj else ""

        metrics = compute_metrics(
            predicted,
            item.get("reference_answer", ""),
            item.get("required_key_points", []),
            item.get("question_type", "GENERAL_SYNTHESIS"),
        )

        row = {
            "question_id": qid,
            "question": item["question"],
            "question_type": item.get("question_type", ""),
            "reference_answer": item.get("reference_answer", ""),
            "predicted_answer": predicted,
            "operator": answer_obj.operator.value if answer_obj and answer_obj.operator else "",
            **metrics,
        }

        if answer_obj and answer_obj.coverage:
            row["coverage_percentage"] = answer_obj.coverage.coverage_percentage

        results.append(row)

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = args.output / f"eval_{mode.value.lower()}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV results: {csv_path}")

    # JSON
    json_path = args.output / f"eval_{mode.value.lower()}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"JSON results: {json_path}")

    # Summary
    total = len(results)
    exact = sum(r.get("exact_match", 0) for r in results)
    print(f"\n{'='*40}")
    print(f"Mode: {mode.value}")
    print(f"Total: {total}")
    print(f"Exact match: {exact}/{total} ({exact/total*100:.1f}%)")


if __name__ == "__main__":
    main()
