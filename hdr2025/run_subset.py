"""
Run a named subset of questions through a pipeline, and nothing else.

Written for testing a change that touches one category. Re-running all 21
questions to see whether three of them improved wastes roughly 85% of the
calls, and on a shared broker key that is 85% of someone else's quota too.

    QIDS=q16,q17,q18 \
    DOC=oecd2026_fullscan_test.txt \
    QUESTIONS=oecd2026_fullscan_questions.json \
    PIPELINE=pipeline_g7.py \
    SUBMISSION_FILE=submission_g7_oecd_arc.json \
    python3 -u run_subset.py

Writes a submission file in the same shape the full runner writes, so the same
scorer reads it unchanged. Questions not named are simply absent from it.

DRY=1 runs the whole thing with the model mocked and spends nothing.
"""

import importlib.util
import json
import os
import sys
import time

PIPELINE = os.environ.get("PIPELINE", "pipeline_g7.py")
DOC = os.environ.get("DOC", "oecd2026_fullscan_test.txt")
QUESTIONS = os.environ.get("QUESTIONS", "oecd2026_fullscan_questions.json")
OUT = os.environ.get("SUBMISSION_FILE", "submission_subset.json")
QIDS = [q.strip() for q in os.environ.get("QIDS", "").split(",") if q.strip()]
DRY = os.environ.get("DRY") == "1"

if not QIDS:
    sys.exit("Set QIDS, e.g. QIDS=q16,q17,q18")

sys.path.insert(0, os.path.dirname(os.path.abspath(PIPELINE)) or ".")
spec = importlib.util.spec_from_file_location("pipeline", PIPELINE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

if DRY:
    calls = {"reader": 0, "synth": 0, "coverage": 0, "gap": 0}

    def fake(prompt: str) -> str:
        # Match on how each prompt OPENS. Matching on content anywhere is what
        # made the first version of this mock reply to the gap-fill prompt as
        # though it were a reader -- the gap prompt embeds the whole evidence
        # block, so it contains every marker a reader reply does, and the mock
        # echoed a slice of its own prompt back into the answer.
        head = prompt.lstrip()[:60]
        if head.startswith("Below is a question"):          # coverage check
            calls["coverage"] += 1
            return "NOTHING MISSING"
        if head.startswith("An answer to the question below"):   # gap fill
            calls["gap"] += 1
            return P.NO_EVIDENCE
        if head.startswith("Write one final answer"):
            calls["synth"] += 1
            return "A mocked final answer."
        calls["reader"] += 1
        if "POSITION: part {n}".replace("{n}", "") in prompt and "STANCE:" in prompt:
            n = prompt.split("TEXT (part ", 1)[1].split(" of", 1)[0]
            return (f"POSITION: part {n}\nSECTION: Chapter {n}\n"
                    f"STANCE: mocked stance.\nFRAMING: mocked\n"
                    f"EVIDENCE: p{n} -- a mocked figure")
        return "Section 1 -- a mocked fact, 42 percent"

    P.call_llm = fake
    P.time.sleep = lambda s: None

questions = {q["id"]: q for q in P.load_questions(QUESTIONS)}
unknown = [q for q in QIDS if q not in questions]
if unknown:
    sys.exit(f"Not in {QUESTIONS}: {', '.join(unknown)}")

print(f"pipeline   {PIPELINE}")
print(f"document   {DOC}")
print(f"answering  {', '.join(QIDS)} ({len(QIDS)} of {len(questions)})"
      + ("   [DRY RUN -- no broker calls]" if DRY else ""))

doc = P.load_document(DOC)
started = time.perf_counter()
answers = []

for qid in QIDS:
    q = questions[qid]
    print(f"\nAnswering {qid} [{q.get('category', 'uncategorized')}]...")
    q_started = time.perf_counter()
    entry = {"id": qid, "question": q["question"], "category": q.get("category")}
    try:
        result = P.answer_question(doc, q["question"], q.get("category"))
        entry.update({"answer": result["answer"], "decision": result["decision"],
                      "readers_with_evidence": result["readers"],
                      "evidence": [f"COMPUTED: {result['computed']}",
                                   f"TEXT EVIDENCE:\n{result['evidence']}"]})
    except Exception as e:                                  # noqa: BLE001
        entry.update({"answer": "", "error": str(e)})
        print(f"  -> failed: {e}")
    entry["seconds"] = round(time.perf_counter() - q_started, 2)
    print(f"  [{entry['seconds']:>7.1f}s] {qid} "
          f"({entry.get('readers_with_evidence', 0)} readers)")
    answers.append(entry)

    # Written after every question, so an interrupted run keeps what it has.
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"team": P.TEAM, "notes": P.NOTES, "answers": answers,
                   "timing": {"subset_of": QIDS, "pipeline": PIPELINE,
                              "document": doc["size"],
                              "answering_seconds": round(time.perf_counter() - started, 2)}},
                  f, indent=2, ensure_ascii=False)

print(f"\nWrote {OUT} ({len(answers)} answers)")
if DRY:
    print(f"  mocked calls: {calls}")
for a in answers:
    print(f"  {a['id']}: {a.get('decision', a.get('error'))}, "
          f"{len(a.get('answer', ''))} chars")
