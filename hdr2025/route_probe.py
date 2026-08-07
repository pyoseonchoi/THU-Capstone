"""
How a question file with NO category field would be routed, and whether the
classifier beats the wording heuristic that fills that gap today.

Step 1 is free. Step 4 costs one ministral-3b-2512 call per question and only
runs with LIVE=1, printing the call count and waiting for YES=1 first.

    python3 -u route_probe.py                 # step 1 only, spends nothing
    LIVE=1 YES=1 python3 -u route_probe.py    # step 1 and step 4

The original dev_questions.yaml is never modified: categories are stripped in
memory only.
"""

import importlib.util
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("G12_FIXES", "headers,arcsection,entityseed,provwindow")

PIPELINE = os.environ.get("PIPELINE", "pipeline_g14.py")
QUESTIONS = os.environ.get("QUESTIONS", "dev_questions.yaml")
LIVE = os.environ.get("LIVE") == "1"
YES = os.environ.get("YES") == "1"


def load(path):
    spec = importlib.util.spec_from_file_location("P", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def questions_of(path):
    here = os.path.join(HERE, path)
    if not os.path.exists(here):
        here = "/Users/yoyomon/Desktop/THU_hackathon/dev_questions.yaml"
    raw = yaml.safe_load(open(here, encoding="utf-8"))
    raw = raw.get("questions", raw) if isinstance(raw, dict) else raw
    return [{"id": q.get("id"), "category": q.get("category"),
             "question": " ".join(str(q["question"]).split())} for q in raw]


def compare(P, qs, use_classifier: bool, label: str):
    os.environ["G14_FIXES"] = (
        os.environ.get("G14_BASE", "answerfield,ratefamily,magnitude,catroute,"
                                   "claimdomain,anchorstop,unitsane")
        + (",classify" if use_classifier else ""))
    P.G14_FIXES = {f.strip() for f in os.environ["G14_FIXES"].split(",") if f.strip()}
    P._CLASSIFIED.clear()
    P.ROUTE_SOURCE.clear()

    wrong = []
    print(f"\n{'=' * 82}\n{label}\n{'=' * 82}")
    print(f"  {'id':<5} {'true category':<18} {'correct mode':<12} "
          f"{'mode with no category':<22} verdict")
    for q in qs:
        right = P.question_mode(q["question"], q["category"])
        got = P.question_mode(q["question"], None)
        via = P.ROUTE_SOURCE.get(q["question"], "?")
        ok = right == got
        if not ok:
            wrong.append((q["id"], q["category"], right, got, via))
        print(f"  {q['id']:<5} {str(q['category']):<18} {right:<12} "
              f"{got + '  (' + via + ')':<22} {'ok' if ok else '*** MISROUTED'}")
    print(f"\n  {len(qs) - len(wrong)}/{len(qs)} correctly routed, "
          f"{len(wrong)} misrouted")
    for i, c, r, g, via in wrong:
        print(f"     {i} [{c}]  correct={r}  got={g}  via {via}")
    return len(wrong)


if __name__ == "__main__":
    P = load(os.path.join(HERE, PIPELINE))
    qs = questions_of(QUESTIONS)

    # Step 1: the baseline risk. No model is called on this path.
    base = compare(P, qs, use_classifier=False,
                   label="STEP 1 -- categories stripped, WORDING HEURISTIC only "
                         "(no model calls)")

    if not LIVE:
        print(f"\n{'=' * 82}")
        print(f"  Step 4 needs {len(qs)} calls on ministral-3b-2512 (one per "
              f"question).")
        print(f"  Re-run with  LIVE=1 YES=1 python3 -u route_probe.py")
        sys.exit(0)
    if not YES:
        print(f"\n  Step 4 would make {len(qs)} calls. Set YES=1 to run it.")
        sys.exit(0)

    # Step 4: the same test through the classifier.
    with_cls = compare(P, qs, use_classifier=True,
                       label=f"STEP 4 -- categories stripped, CLASSIFIER first "
                             f"({len(qs)} calls on {P.CHEAP_MODEL})")

    print(f"\n{'=' * 82}\nVERDICT\n{'=' * 82}")
    print(f"  wording heuristic alone : {len(qs) - base}/{len(qs)} correct "
          f"({base} misrouted)")
    print(f"  classifier first        : {len(qs) - with_cls}/{len(qs)} correct "
          f"({with_cls} misrouted)")
    if with_cls < base:
        print(f"\n  The classifier corrects {base - with_cls} question(s). KEEP IT.")
    elif with_cls == base:
        print(f"\n  No improvement. The extra call buys nothing -- DROP IT "
              f"(G14_FIXES without 'classify').")
    else:
        print(f"\n  The classifier is WORSE by {with_cls - base}. DROP IT.")
