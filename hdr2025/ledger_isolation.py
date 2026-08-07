"""
Prove that g13 changes the contradiction path and NOTHING else.

The brief names four categories as controls -- needle, absence, global
synthesis, cross-section -- and notes that two supposedly-scoped changes have
regressed a control before. A scored re-run cannot settle that: the same code
over the same data has scored 67.0% and 60.4%.

What settles it: g12 and g13, handed the same document and the same question,
build the same reader and synthesis prompts for every question EXCEPT the
contradiction ones. Prompts are the entire interface to the broker, and the
synthesis prompt embeds the computed block, so identical prompts mean identical
inputs AND identical Python arithmetic upstream of them.

Every model call is mocked. This costs nothing.

    python3 -u ledger_isolation.py
"""

import importlib.util
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("G12_FIXES", "headers,arcsection,entityseed,provwindow")

BASE = os.environ.get("BASE", "pipeline_g13.py")
NEW = os.environ.get("NEW", "pipeline_g14.py")

CASES = [
    ("PARKS", "nationalparks_europe.txt", "dev_questions.yaml",
     "/Users/yoyomon/Desktop/THU_hackathon/nationalparks_europe.txt",
     "/Users/yoyomon/Desktop/THU_hackathon/dev_questions.yaml"),
    ("OECD", "oecd2026_fullscan_test.txt", "oecd2026_fullscan_questions.json",
     "/Users/yoyomon/Downloads/oecd2026_fullscan_test.txt",
     "/Users/yoyomon/Downloads/oecd2026_fullscan_questions.json"),
    ("GEM", "gem2024_5_fullscan_no_indent.txt", "gem2024_5_expected_questions.json",
     "/Users/yoyomon/Downloads/gem2024_5_fullscan_no_indent.txt",
     "/Users/yoyomon/Downloads/gem2024_5_expected_questions.json"),
]

CONTRADICTION = {"contradiction", "aggregation", "superlative", "cross_section"}


def resolve(local, fallback):
    here = os.path.join(HERE, local)
    return here if os.path.exists(here) else fallback


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def instrument(P):
    sent = []
    marker = "TEXT (part "

    def fake(prompt: str, model: str | None = None) -> str:
        sent.append(prompt)
        head = prompt.lstrip()[:60]
        if head.startswith("Below is a question"):
            return "NOTHING MISSING"
        if head.startswith("An answer to the question below"):
            return P.NO_EVIDENCE
        if head.startswith("Below are an answer"):
            return "A mocked merged answer, 42 percent."
        if head.startswith("Below is a shortlist"):
            return "1 | SAME | mocked"
        if marker not in prompt:
            return "A mocked final answer."
        if "ANCHOR:" in prompt:
            return ("ANCHOR: mocked phrase\nENTITY: Mockland\n"
                    "FACT: a mocked fact, 42 percent.\nWHERE: Chapter 1")
        if "STANCE:" in prompt:
            part = prompt.split(marker, 1)[1].split(" of", 1)[0]
            return (f"POSITION: part {part}\nSECTION: Chapter {part}\n"
                    f"STANCE: mocked.\nFRAMING: mocked\nEVIDENCE: p{part} -- x")
        return "Section 1 -- a mocked fact, 42 percent"

    P.call_llm = fake
    P.time.sleep = lambda s: None
    return sent


def run(P, doc_path, questions_path):
    doc = P.load_document(doc_path)
    questions = P.load_questions(questions_path)
    out = {}
    for q in questions:
        sent = instrument(P)
        try:
            P.answer_question(doc, q["question"], q.get("category"))
        except Exception as e:                                  # noqa: BLE001
            sent.append(f"__EXCEPTION__ {type(e).__name__}: {e}")
        out[q["id"]] = (q.get("category") or "?", list(sent))
    return out


def compare(label, doc_path, questions_path):
    A = run(load("BASE", os.path.join(HERE, BASE)), doc_path, questions_path)
    B = run(load("NEW", os.path.join(HERE, NEW)), doc_path, questions_path)

    per_cat = defaultdict(lambda: [0, 0, 0, 0])     # same, diff, callsA, callsB
    offenders = []
    for qid, (cat, a) in sorted(A.items()):
        _, b = B.get(qid, (cat, []))
        row = per_cat[cat]
        row[2] += len(a)
        row[3] += len(b)
        if a == b:
            row[0] += 1
        else:
            row[1] += 1
            if cat not in CONTRADICTION:
                offenders.append((qid, cat, len(a), len(b)))

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"  {'category':<18} {'identical':>9} {'differ':>7} "
          f"{'calls ' + BASE[9:12]:>11} {'calls ' + NEW[9:12]:>11}  verdict")
    ok = True
    for cat in sorted(per_cat):
        same, diff, ca, cb = per_cat[cat]
        expected_to_differ = cat in CONTRADICTION
        good = (diff == 0) if not expected_to_differ else True
        ok = ok and good
        verdict = ("CHANGED (intended)" if expected_to_differ and diff
                   else "unchanged" if diff == 0
                   else "*** LEAKED ***")
        print(f"  {cat:<18} {same:>9} {diff:>7} {ca:>11} {cb:>11}  {verdict}")
    total_a = sum(v[2] for v in per_cat.values())
    total_b = sum(v[3] for v in per_cat.values())
    print(f"  {'TOTAL':<18} {'':>9} {'':>7} {total_a:>11} {total_b:>11}  "
          f"delta {total_b - total_a:+d}")
    if offenders:
        print("\n  LEAKS -- these are controls and must not have moved:")
        for qid, cat, ca, cb in offenders:
            print(f"     {qid} [{cat}] {ca} vs {cb} prompts")
    return ok, offenders


if __name__ == "__main__":
    print("Every model call is mocked; this run spends nothing.")
    results = {}
    for label, local_doc, local_q, fb_doc, fb_q in CASES:
        doc, qs = resolve(local_doc, fb_doc), resolve(local_q, fb_q)
        if not (os.path.exists(doc) and os.path.exists(qs)):
            print(f"\nskipping {label}: files not found")
            continue
        ok, _ = compare(f"{label}: {BASE} vs {NEW}", doc, qs)
        results[label] = ok

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    for k, v in results.items():
        print(f"  {k:<8} controls untouched: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
