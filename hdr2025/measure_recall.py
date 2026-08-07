"""
Measure extraction recall without answering anything, and without guessing.

Two modes. Prefer the first.

  REPLAY (free).  A finished submission file already contains, per question, the
  reader lines that produced its answer. Re-parse those lines and push them
  through both pipelines' arithmetic. This measures the ONE thing we cannot see
  from a score report: how many records were extracted versus how many survived
  the code between extraction and the answer.

      SUBMISSION=submission_g8b_parks.json python3 -u measure_recall.py

  LIVE (metered).  Run the record-extraction stage over the document and nothing
  else -- no synthesis, no coverage pass, no gap fill. Costs one model call per
  slice per checklist question. It prints the call count and waits for
  confirmation before spending anything.

      LIVE=1 DOC=nationalparks_europe.txt python3 -u measure_recall.py

Records are written to disk as they are produced, so an interrupted run still
leaves something to read.
"""

import importlib.util
import json
import os
import re
import sys

PIPELINE = os.environ.get("PIPELINE", "pipeline_g9.py")
BASELINE = os.environ.get("BASELINE", "pipeline_g8.py")
SUBMISSION = os.environ.get("SUBMISSION", "submission_g8b_parks.json")
DOC = os.environ.get("DOC", "nationalparks_europe.txt")
OUT = os.environ.get("RECALL_OUT", "recall_records.json")
LIVE = os.environ.get("LIVE") == "1"
YES = os.environ.get("YES") == "1"


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ GROUND TRUTH
#
# Taken from dev_questions.yaml by hand and frozen here, so that this harness
# never reads the answer key at runtime and cannot leak it into a prompt.
# Names are matched on the pipeline's own _entity_key, so 'Ordesa' and 'Ordesa
# National Park' both count.

CHECKLISTS = {
    "d02 highest point >= 3000 m (8 parks)": [
        "Ecrins", "Hohe Tauern", "Sierra Nevada", "Etna",
        "Ordesa", "Pyrenees", "Swiss", "Aiguestortes",
    ],
    "d01 parks in Spain (6 parks)": [
        "Aiguestortes", "Atlantic Islands of Galicia", "Donana",
        "Ordesa", "Picos de Europa", "Sierra Nevada",
    ],
    "d03 parks in Croatia (3 parks)": [
        "Kornati", "Paklenica", "Plitvice",
    ],
}


def _norm(name: str, P) -> str:
    """Fold a name the way the pipeline folds it, minus accents."""
    key = P._entity_key(name or "")
    return (key.replace("é", "e").replace("è", "e")
               .replace("ü", "u").replace("ñ", "n")
               .replace("ç", "c").replace("â", "a")
               .replace("í", "i").replace("ó", "o"))


def report_checklists(records, P) -> None:
    found = {_norm(r["entity"], P) for r in records}
    for title, wanted in CHECKLISTS.items():
        hits = [w for w in wanted
                if any(_norm(w, P) and _norm(w, P) in f for f in found)]
        pct = 100.0 * len(hits) / len(wanted)
        print(f"  {title}")
        print(f"    extracted {len(hits)}/{len(wanted)} = {pct:.0f}%")
        missing = [w for w in wanted if w not in hits]
        if missing:
            print(f"    missing:  {', '.join(missing)}")


# ------------------------------------------------------------------ REPLAY

_READER = re.compile(r"^READER\s+(\d+)\s*:", re.M)


def replay(P) -> dict:
    """Pull every RECORD/CLAIM line back out of a finished submission."""
    with open(SUBMISSION, encoding="utf-8") as f:
        submission = json.load(f)

    print(f"replaying  {SUBMISSION}")
    print(f"notes      {submission.get('notes', '')[:300]}")
    print()

    per_question = {}
    for answer in submission.get("answers", []):
        blob = "\n".join(str(x) for x in (answer.get("evidence") or []))
        if not blob.strip():
            continue
        # Split on the READER n: headers so each record keeps the slice that
        # produced it -- the ledger's 'separated' test depends on it.
        pieces, last, name = [], 0, "unknown"
        for m in _READER.finditer(blob):
            pieces.append((name, blob[last:m.start()]))
            name, last = f"reader {m.group(1)}", m.end()
        pieces.append((name, blob[last:]))

        records, claims = [], []
        for source, chunk in pieces:
            records += P.parse_records(chunk, source=source)
            claims += P.parse_claims(chunk, source=source)
        per_question[answer["id"]] = {
            "category": answer.get("category"),
            # Carried through because parse_threshold reads it: without the
            # question, '>= 3000 m' is invisible and every record "matches".
            "question": answer.get("question", ""),
            "records": records, "claims": claims,
            "computed_was_empty": "nothing could be computed" in blob,
        }
    return per_question


def summarise(per_question, P, label) -> list:
    all_records = []
    print(f"================ {label} ================")
    for qid, data in sorted(per_question.items()):
        records = data["records"]
        all_records += records
        entities = {_norm(r["entity"], P) for r in records}
        flag = "  <- COMPUTED WAS EMPTY" if data["computed_was_empty"] else ""
        print(f"  {qid:<5} [{(data['category'] or '?'):<17}] "
              f"{len(records):>4} records, {len(entities):>3} entities, "
              f"{len(data['claims']):>3} claims{flag}")
    print(f"\n  TOTAL {len(all_records)} records, "
          f"{len({_norm(r['entity'], P) for r in all_records})} distinct entities\n")
    report_checklists(all_records, P)
    return all_records


def compare_arithmetic(per_question, G8, G9) -> None:
    """
    The measurement that matters: same records, two versions of the code that
    turns records into an answer.
    """
    print("\n================ SAME RECORDS, TWO TALLIES ================")
    print("  (extraction is identical in both columns -- only the arithmetic differs)\n")
    for qid, data in sorted(per_question.items()):
        if not data["records"]:
            continue
        question = data.get("question", "")
        row = [qid]
        for M in (G8, G9):
            records = [M._make_record(r["entity"], r["attribute"], r["raw_value"],
                                      r["where"], r["source"], r["kind"])
                       for r in data["records"]]
            out = M.tally(records, question)
            if not out:
                row.append("-")
            elif "matching_count" in out:
                row.append(f"{out['matching_count']} of {out['distinct_entities']}"
                           f" [{out['attribute_ranked'][:22]}]")
            else:
                row.append(f"groups {out.get('group_counts')}")
        if row[1] != row[2]:
            print(f"  {row[0]:<5} g8: {row[1]:<44} g9: {row[2]}")
    print()


# -------------------------------------------------------------------- LIVE

def live(P) -> dict:
    doc = P.load_document(DOC)
    text = P.strip_back_matter(doc["text"]) if P.step_on("6") else doc["text"]
    chunks = P.chunk_lines(text, P.WORDS_PER_READER)
    questions = [
        "Which parks report a highest point, and what is it?",
        "Which country is each park in?",
    ]
    calls = len(chunks) * len(questions)
    print(f"document   {DOC} ({doc['size']})")
    print(f"slices     {len(chunks)}")
    print(f"THIS WILL MAKE {calls} MODEL CALLS and nothing else.")
    if not YES:
        print("Set YES=1 to actually run it. Nothing was spent.")
        sys.exit(0)

    per_question = {}
    for qi, question in enumerate(questions):
        jobs = [(f"reader {i + 1}",
                 P.RECORD_PROMPT.format(n=i + 1, of=len(chunks), doc=doc["name"],
                                        scope="", marker=P.NO_EVIDENCE,
                                        chunk=c["text"], question=question))
                for i, c in enumerate(chunks)]
        replies, _ = P._run_readers(jobs)
        records, claims = [], []
        for source, reply in replies.items():
            records += P.parse_records(reply, source=source)
            claims += P.parse_claims(reply, source=source)
        per_question[f"live{qi + 1}"] = {"category": "live", "question": question,
                                         "records": records, "claims": claims,
                                         "computed_was_empty": False}
        with open(OUT, "w", encoding="utf-8") as f:      # incremental, always
            json.dump(per_question, f, indent=2, ensure_ascii=False)
    return per_question


# -------------------------------------------------------------------- MAIN

if __name__ == "__main__":
    G9 = load("G9", PIPELINE)
    per_question = live(G9) if LIVE else replay(G9)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({q: {"category": d["category"], "records": d["records"],
                       "claims": d["claims"]}
                   for q, d in per_question.items()}, f, indent=2,
                  ensure_ascii=False)

    summarise(per_question, G9, "EXTRACTION (what the readers actually returned)")

    if os.path.exists(BASELINE):
        compare_arithmetic(per_question, load("G8", BASELINE), G9)

    print(f"every record dumped to {OUT}")
