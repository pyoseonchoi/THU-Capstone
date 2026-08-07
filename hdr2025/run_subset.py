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
    # A plain dict here meant calls["adjudicate"] += 1 raised KeyError inside
    # the pipeline's own try/except, which reported it as "adjudication failed"
    # -- a mock defect wearing a pipeline defect's clothes.
    from collections import defaultdict
    calls = defaultdict(int)

    # A READER prompt is the only kind that carries the verbatim slice, and it
    # always announces it the same way. Every other prompt embeds the readers'
    # REPLIES instead, which is why matching on reply markers cannot work:
    # the arc synthesis prompt contains 'POSITION: part' and 'STANCE:' because
    # the reader lines are quoted inside it, and it contains no slice at all.
    # An earlier mock read those markers, took the reader branch, and tried to
    # split out a part number that was not there -- so the pipeline's own
    # try/except reported 'synthesis failed (list index out of range)'. A mock
    # defect wearing a pipeline defect's clothes, and the second time this file
    # has produced one.
    #
    # So: identify a reader by the slice marker, positively, and let everything
    # else be a synthesis-shaped prompt. Nothing here indexes into a split.
    _READER_MARKER = "TEXT (part "

    def fake(prompt: str, model: str | None = None) -> str:
        head = prompt.lstrip()[:60]
        if head.startswith("Below is a question"):          # coverage check
            calls["coverage"] += 1
            return "NOTHING MISSING"
        if head.startswith("An answer to the question below"):   # gap fill
            calls["gap"] += 1
            return P.NO_EVIDENCE
        if head.startswith("Below are an answer"):          # g8 merge
            calls["merge"] += 1
            return "A mocked merged answer, 42 percent."
        if head.startswith("Below is a shortlist"):         # g8 adjudication
            calls["adjudicate"] += 1
            return "1 | CONFLICT | two tables give different values"

        if _READER_MARKER not in prompt:
            # No slice: this is a synthesis, however it happens to open. Counted
            # under its opening words so an unrecognised prompt is VISIBLE in
            # the tally rather than silently answered as something else.
            known = (head.startswith("Write one final answer")
                     or head.startswith("Below are reports from readers")
                     or head.startswith("Answer the question below"))
            calls["synth" if known else f"synth?({head[:28]!r})"] += 1
            return "A mocked final answer."

        calls["reader"] += 1
        if "ANCHOR:" in prompt:                             # g8 cross-section
            return ("ANCHOR: mocked phrase\nENTITY: Mockland\n"
                    "FACT: a mocked fact, 42 percent.\nWHERE: Chapter 1")
        if "STANCE:" in prompt:                             # arc reader
            part = prompt.split(_READER_MARKER, 1)[1].split(" of", 1)[0]
            return (f"POSITION: part {part}\nSECTION: Chapter {part}\n"
                    f"STANCE: mocked stance.\nFRAMING: mocked\n"
                    f"EVIDENCE: p{part} -- a mocked figure")
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

# The one line that says which path is live. Worth printing here too: a subset
# run is where a change gets checked, and three generations shipped with a
# subsystem switched off because nobody read the log for it.
if "chunking_mode" in doc:
    print(f"segmentation  {len(doc['headers']):,} '## ' headers, coverage "
          f"{doc['header_coverage']:.3f} -> chunking_mode = {doc['chunking_mode']}"
          + (f"; entity roster {len(doc['entity_roster'])}"
             if doc.get("entity_roster") else "; no entity roster")
          + (f"; provenance window {P.provenance_window(doc)}"
             if hasattr(P, "provenance_window") else ""))

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
