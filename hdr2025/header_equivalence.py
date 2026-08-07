"""
Prove that g12's fallback path is g11, byte for byte, without spending a call.

A scored re-run cannot prove this. Two runs of the same code over the same data
scored 67.0% and 60.4% once (handoff section 6), so "OECD came back about the
same" is not evidence of anything. What IS evidence: g11 and g12, handed the
same document and the same question, build the IDENTICAL set of reader prompts
and the identical chunk boundaries. If every byte the broker would receive is
the same, the answer distribution is the same by construction.

The model is mocked throughout, so this costs nothing.

    python3 -u header_equivalence.py
"""

import importlib.util
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Everything is looked for beside this file, so the whole set can be copied
# into one folder and run there. A pair that is not present is skipped.
DOCS = {k: os.path.join(HERE, v) for k, v in {
    "parks": "nationalparks_europe.txt",
    "oecd": "oecd2026_fullscan_test.txt",
    "gem": "gem2024_5_fullscan_no_indent.txt",
    "hdr": "hdr2025.txt",
}.items()}
QUESTIONS = {k: os.path.join(HERE, v) for k, v in {
    "parks": "dev_questions.yaml",
    "oecd": "oecd2026_fullscan_questions.json",
    "gem": "gem2024_5_expected_questions.json",
    "hdr": "hdr2025_fullscan_questions.json",
}.items()}


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instrument(P):
    """Capture every prompt the pipeline would send, and answer it cheaply."""
    sent = []

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
        if (head.startswith("Write one final answer")
                or head.startswith("Below are reports from readers")
                or head.startswith("Answer the question below")):
            return "A mocked final answer."
        if "ANCHOR:" in prompt:
            return ("ANCHOR: mocked phrase\nENTITY: Mockland\n"
                    "FACT: a mocked fact, 42 percent.\nWHERE: Chapter 1")
        if "STANCE:" in prompt:
            return ("POSITION: part 1\nSECTION: Chapter 1\nSTANCE: mocked.\n"
                    "FRAMING: mocked\nEVIDENCE: p1 -- a mocked figure")
        return "Section 1 -- a mocked fact, 42 percent"

    P.call_llm = fake
    P.time.sleep = lambda s: None
    return sent


def run_one(P, doc_path, questions_path):
    """Every prompt this pipeline builds for every question, keyed by id."""
    doc = P.load_document(doc_path)
    questions = P.load_questions(questions_path)
    per_q = {}
    for q in questions:
        sent = instrument(P)
        try:
            P.answer_question(doc, q["question"], q.get("category"))
        except Exception as e:                                  # noqa: BLE001
            sent.append(f"__EXCEPTION__ {type(e).__name__}: {e}")
        per_q[q["id"]] = list(sent)
    return doc, per_q


def compare(label, doc_path, questions_path, g12_fixes):
    os.environ["G12_FIXES"] = g12_fixes
    G11 = load("G11", os.path.join(HERE, "pipeline_g11.py"))
    G12 = load("G12", os.path.join(HERE, "pipeline_g12.py"))

    d11, a = run_one(G11, doc_path, questions_path)
    d12, b = run_one(G12, doc_path, questions_path)

    print(f"\n{'=' * 74}")
    print(f"{label}   (G12_FIXES={g12_fixes!r})")
    print(f"{'=' * 74}")
    print(f"  g12 chunking_mode: {d12['chunking_mode']}   "
          f"headers: {len(d12['headers']):,}   "
          f"coverage: {d12['header_coverage']:.3f}   "
          f"roster: {len(d12['entity_roster'])}")
    print(f"  slices  g11: {len(d11['chunks'])}    g12: {len(d12['chunks'])}")

    same_q, diff_q = [], []
    calls = defaultdict(lambda: [0, 0])
    for qid in sorted(a):
        calls[qid] = [len(a[qid]), len(b.get(qid, []))]
        if a[qid] == b.get(qid):
            same_q.append(qid)
        else:
            diff_q.append(qid)

    total11 = sum(v[0] for v in calls.values())
    total12 = sum(v[1] for v in calls.values())
    print(f"  model calls  g11: {total11}    g12: {total12}    "
          f"delta: {total12 - total11:+d}")
    print(f"  questions whose prompts are BYTE-IDENTICAL: "
          f"{len(same_q)}/{len(a)}")
    if diff_q:
        print(f"  questions that differ: {', '.join(diff_q)}")
        for qid in diff_q[:3]:
            x, y = a[qid], b.get(qid, [])
            print(f"    {qid}: {len(x)} vs {len(y)} prompts")
            for i, (p, q) in enumerate(zip(x, y)):
                if p != q:
                    print(f"      first difference at prompt {i}: "
                          f"{len(p)} vs {len(q)} chars")
                    break
    return len(diff_q) == 0, total11, total12


ALL = "headers,arcsection,entityseed,provwindow"

if __name__ == "__main__":
    print("Every model call is mocked; this run spends nothing.\n")
    verdicts = {}

    # 1. The documents with no headers must be identical with g12 FULLY ON.
    for key in ("oecd", "gem", "hdr"):
        if not os.path.exists(DOCS[key]) or not os.path.exists(QUESTIONS[key]):
            print(f"skipping {key}: file not found")
            continue
        ok, t11, t12 = compare(f"{key.upper()} -- header-absent, g12 fully on",
                               DOCS[key], QUESTIONS[key],
                               ALL)
        verdicts[key] = ok

    # 2. The header-carrying document must be identical when g12 is switched OFF.
    if not os.path.exists(DOCS["parks"]):
        sys.exit("nationalparks_europe.txt is not beside this file; nothing to compare")
    ok, t11, t12 = compare("PARKS -- header-rich, g12 switched OFF",
                           DOCS["parks"], QUESTIONS["parks"], "")
    verdicts["parks_off"] = ok

    # 3. And it must actually change when g12 is on -- otherwise nothing shipped.
    changed, t11, t12 = compare("PARKS -- header-rich, g12 fully on",
                                DOCS["parks"], QUESTIONS["parks"],
                                ALL)
    verdicts["parks_on_changes"] = not changed

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    for k, v in verdicts.items():
        expect = ("must differ" if k == "parks_on_changes" else "must match")
        print(f"  {k:<20} {expect:<12} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(verdicts.values()) else 1)
