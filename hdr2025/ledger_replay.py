"""
Replay a finished run's reader lines through g13's candidate assembly.

Same records, same claims, two versions of the code between extraction and the
answer. Every model call is skipped, so this costs nothing and is exactly
reproducible -- which a scored run is not (67.0% and 60.4% on the same code
over the same data, handoff section 6).

    python3 -u ledger_replay.py
    G13_FIXES=textclaims python3 -u ledger_replay.py     # bisect one flag
"""

import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("G12_FIXES", "headers,arcsection,entityseed,provwindow")

PIPELINE = os.environ.get("PIPELINE", "pipeline_g13.py")
BASELINE = os.environ.get("BASELINE", "pipeline_g12.py")
SUBMISSION = os.environ.get(
    "SUBMISSION", "/Users/yoyomon/Downloads/submission_g12_parks-2.json")
DOC = os.environ.get("DOC", "nationalparks_europe.txt")
QIDS = [q.strip() for q in os.environ.get("QIDS", "d10,d11,d12").split(",") if q.strip()]

# What the frozen answer key wants, for reporting only. Never read by the code
# under test and never put into a prompt.
WANTED = {
    "d10": ("snowdon", "1085", "macdui", "1309"),
    "d11": ("hardangervidda", "3430", "spitsbergen", "3583"),
    "d12": ("jotunheimen", "1151", "mile"),
}

_READER = re.compile(r"^READER\s+(.+?):", re.M)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def reader_pieces(answer):
    blob = "\n".join(str(x) for x in answer.get("evidence", []))
    pieces, last, name = [], 0, "unknown"
    for m in _READER.finditer(blob):
        pieces.append((name, blob[last:m.start()]))
        name, last = m.group(1), m.end()
    pieces.append((name, blob[last:]))
    return pieces


def assemble(P, answer, doc, question):
    """Everything the ledger branch does, up to the computed block."""
    against = P.strip_back_matter(doc["text"])
    window = P.provenance_window(doc)
    pieces = reader_pieces(answer)
    records = [r for s, c in pieces for r in P.parse_records(c, source=s)]
    claims = [c for s, c2 in pieces for c in P.parse_claims(c2, source=s)]
    verified = [r for r in records
                if P.verify_record(r, against, window).startswith("verified")
                or P.verify_record(r, against, window) == "unverifiable"]
    names = sorted({(r.get("entity") or "").strip() for r in verified},
                   key=len, reverse=True)

    volunteered = len(claims)
    group_claims = []
    if hasattr(P, "find_text_claims") and P.g13_on("textclaims"):
        claims = claims + P.find_text_claims(against, names)
        group_claims = P.find_group_rank_claims(against, names)

    candidates = (P.find_value_conflicts(verified)
                  + P.find_unit_outliers(verified)
                  + P.check_claims(claims, verified))
    if hasattr(P, "find_rank_conflicts") and P.g13_on("rank"):
        candidates = P.find_rank_conflicts(claims, group_claims, verified, against) + candidates
    if P.g10_on("labelunit"):
        candidates = candidates + P.find_label_unit_outliers(against, names)

    chosen = None
    if hasattr(P, "select_one_conflict") and P.g13_on("onepair"):
        chosen = P.select_one_conflict(candidates, question)
    return {"records": len(records), "verified": len(verified),
            "claims_volunteered": volunteered, "claims_total": len(claims),
            "group_claims": len(group_claims), "candidates": candidates,
            "chosen": chosen}


def report(label, P, data, questions, doc):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for qid in QIDS:
        answer = next((a for a in data["answers"] if a["id"] == qid), None)
        if not answer:
            continue
        q = questions.get(qid, "")
        out = assemble(P, answer, doc, q)
        print(f"\n  {qid}: {out['records']} records -> {out['verified']} verified | "
              f"claims {out['claims_volunteered']} volunteered"
              + (f" + {out['claims_total'] - out['claims_volunteered']} from text"
                 if out["claims_total"] > out["claims_volunteered"] else "")
              + (f" | {out['group_claims']} group-rank" if out["group_claims"] else "")
              + f" | {len(out['candidates'])} candidates")
        chosen = out["chosen"]
        if not chosen:
            print(f"     SELECTED: none -- answer would state 'no verified conflict'")
            continue
        print(f"     SELECTED: {chosen['kind']}"
              + (f"/{chosen.get('test')}" if chosen.get("test") else "")
              + f"  (route score {P.route_score(chosen, q)})")
        sentence = chosen["comparison_sentence"]
        for i in range(0, len(sentence), 92):
            print(f"       {sentence[i:i + 92]}")
        want = WANTED.get(qid, ())
        low = sentence.lower()
        hit = [w for w in want if w in low]
        print(f"     key terms present: {len(hit)}/{len(want)}  "
              f"{'ALL PRESENT' if len(hit) == len(want) else 'missing ' + str([w for w in want if w not in low])}")


if __name__ == "__main__":
    data = json.load(open(SUBMISSION, encoding="utf-8"))
    qtext = {}
    import yaml
    for q in yaml.safe_load(open(os.path.join(HERE, "dev_questions.yaml")
                                 if os.path.exists(os.path.join(HERE, "dev_questions.yaml"))
                                 else "/Users/yoyomon/Desktop/THU_hackathon/dev_questions.yaml")):
        qtext[q["id"]] = " ".join(str(q["question"]).split())

    doc_path = DOC if os.path.exists(DOC) else os.path.join(HERE, DOC)
    if not os.path.exists(doc_path):
        doc_path = "/Users/yoyomon/Desktop/THU_hackathon/nationalparks_europe.txt"

    NEW = load("NEW", os.path.join(HERE, PIPELINE))
    doc = NEW.load_document(doc_path)
    if os.path.exists(os.path.join(HERE, BASELINE)):
        OLD = load("OLD", os.path.join(HERE, BASELINE))
        report(f"BEFORE -- {BASELINE} (candidates go to the adjudicator as a pile)",
               OLD, data, qtext, doc)
    report(f"AFTER -- {PIPELINE} (G13_FIXES={','.join(sorted(NEW.G13_FIXES))})",
           NEW, data, qtext, doc)
