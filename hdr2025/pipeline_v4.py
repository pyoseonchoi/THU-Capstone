"""
Model D: full park entries -- table AND prose, with validated figures.

Scoring v3 gave 47.0% (up from v2's 38.6%), but it traded one weakness for
another:

    v2 -> v3   Aggregation 0 -> 47 | Absence 28 -> 64 | Contradiction 17 -> 39
               Needle 83 -> 64 | Cross-section 33 -> 0

The table fixed counting and broke everything that needs prose. Diagnosing the
individual zeros on the dev set:

  d04 (largest park) failed because the boxes are scrambled by PDF conversion
      and the cheap model mispaired numbers with labels. Box 24 reads
      "820 Area covered (sq km) 135,000 Pea..." -- the area is 820; the 135,000
      belongs to the NEXT stat. Read as 135,000 it beats the true largest
      (Vatnajokull, 13,600 sq km) and the answer is wrong.
  d03 (Croatia/Montenegro) failed because country was guessed from loose prose
      mentions: "Croatia" appears near 6 boxes and "Montenegro" near 4, and
      they overlap -- a Montenegrin park reached via Croatia mentions both.
  d13/d14/d15 (cross-section) and d19/d20 (needle) failed because v3 only ever
      read the 1200 words BEFORE each box. The Icehotel, the Unesco
      inscriptions and the climbing firsts are all in the prose AFTER it.

So v4 changes three things:

  1. ENTRY SPANS, not lookback windows. Park i runs from 1200 words before its
     box to 1200 words before the next one, which captures the whole entry --
     heading, intro, box, and the "Stay here / Do this / What to spot" prose.
  2. RICHER EXTRACTION on a stronger model: alongside the stats, each park
     yields its designations, cross-border status, named features and dated
     notable facts. Country now requires quoted evidence.
  3. PLAUSIBILITY VALIDATION in Python. A figure outside a sane range is
     flagged and re-extracted from the box alone. This is what catches the
     135,000 mispairing without anyone having to read 60 boxes by hand.

Python still owns all counting and ranking; the model never counts.

Requires:
    pip install openai pyyaml --break-system-packages
    export BROKER_API_KEY=your_group_key
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline_ed2 import (
    call_llm,
    call_llm_json,
    LLMJSONError,
    preprocess_document,
    load_questions,
)
from pipeline_v3 import (
    BOX_ANCHOR,
    parse_number,
    normalize_park,
    compute_aggregates,
)

MODEL_EXTRACT = "mistral-small-3-2"   # v3 used the 3b model and mispaired figures
MODEL_ANSWER = "mistral-small-3-2"

# A European national park outside these bounds means the box was misread, not
# that the park is remarkable. Vatnajokull, the largest here, is 13,600 sq km.
PLAUSIBLE_AREA_SQ_KM = (0.5, 20_000)
PLAUSIBLE_HEIGHT_M = (0, 5_000)
PLAUSIBLE_VISITORS = (0, 100_000_000)


# --------------------------------------------------- A. SPLIT INTO ENTRIES

def find_park_entries(text: str, lookback: int = 1200) -> list[dict]:
    """
    Cut the document into one span per park.

    Each entry runs heading -> intro -> Toolbox -> Getting there -> box ->
    Stay here -> Do this -> What to spot -> Hike this -> Itineraries, and the
    box sits roughly 1200 words in. So anchoring on the box and reaching back
    1200 words lands on the heading, and running forward to 1200 words before
    the NEXT box captures everything after it.

    v3 stopped at the box and lost every fact in the second half of an entry.
    """
    words = text.split()
    anchors = [
        i for i in range(len(words) - 2)
        if words[i] == "Park" and words[i + 1] == "in" and words[i + 2] == "numbers"
    ]

    entries = []
    for n, anchor in enumerate(anchors):
        start = max(0, anchor - lookback)
        if n + 1 < len(anchors):
            end = max(start + 1, anchors[n + 1] - lookback)
        else:
            end = len(words)

        # The box itself, verbatim, so the extractor sees it undiluted too.
        box_words = words[anchor:anchor + 60]
        entries.append({
            "index": n + 1,
            "entry": " ".join(words[start:end]),
            "box": " ".join(box_words),
        })
    return entries


# ------------------------------------------------------- B. EXTRACT A PARK

EXTRACT_PROMPT = """You are reading one complete park entry from a book of European national parks.

Extract ONLY what this text states. Never infer from general knowledge -- if
the text does not say it, use null or an empty list.

THE NUMBERS BOX IS SCRAMBLED. PDF conversion separated numbers from their
labels, and a number sometimes appears after the label it belongs to, or
before the previous one. Pair them carefully.
Sanity check every pairing: a national park's area is between 1 and 15,000 sq
km, and a European summit is under 5,000 m. If a pairing gives an area of
135,000 or a summit of 30,000, you have attached the wrong number -- that
figure belongs to a different stat in the same box (an age, a species count,
a number of visitors).

For "country", quote the exact sentence you took it from in
"country_evidence". Beware: entries often mention neighbouring countries for
travel directions ("accessible via Croatia"). The country is where the PARK
is, not where you drive from.

Return valid JSON, no prose, matching exactly this schema:

{{
  "name": "the park's name as the book writes it",
  "country": "country the park is in, or null if not stated",
  "country_evidence": "the exact sentence stating the country, or null",
  "stats": [
    {{"label": "label exactly as printed, e.g. 'Area covered (sq km)'",
      "value": "the number exactly as printed, e.g. '1,151' or '370-550'"}}
  ],
  "designations": ["e.g. 'Unesco World Heritage Site', 'Unesco Biosphere Reserve', 'Natura 2000'"],
  "cross_border": {{
    "is_transboundary": true or false,
    "partner_countries": ["countries the park crosses into or is paired with"],
    "detail": "what the text says about the border, or null"
  }},
  "named_features": ["named peaks, lakes, trails, glaciers, hotels, villages"],
  "notable_facts": [
    "one sentence per specific fact, KEEPING any date, year, number or name",
    "include firsts and foundings, e.g. 'X was first climbed in 1865'",
    "include named places to stay and what they are, e.g. 'The Icehotel started as a small igloo-art gallery in 1989'",
    "include wildlife, conservation status and threats named in the text"
  ]
}}

PARK IN NUMBERS BOX (verbatim):
{box}

FULL ENTRY TEXT:
{entry}
"""

RECHECK_PROMPT = """This "Park in numbers" box was misread: the extracted figures are outside any
plausible range for a national park, which means a number was paired with the
wrong label.

The box is scrambled by PDF conversion. Each number belongs to exactly one
label. A park's area is between 1 and 15,000 sq km; a European summit is under
5,000 m. Large round numbers like 135,000 or 30,000 are almost always an age,
a species count or a visitor count -- not an area.

Previous (wrong) reading: {previous}

BOX (verbatim):
{box}

Return valid JSON, no prose:

{{"stats": [{{"label": "label exactly as printed", "value": "the number"}}]}}
"""


def extract_park(entry: dict) -> dict:
    """Extract one park's full record. Never raises."""
    prompt = EXTRACT_PROMPT.format(box=entry["box"], entry=entry["entry"])
    try:
        record = call_llm_json(
            prompt, MODEL_EXTRACT, label=f"park {entry['index']}"
        )
    except LLMJSONError as e:
        print(f"  -> park {entry['index']} extraction failed: {e}")
        record = {}

    record["index"] = entry["index"]
    record["raw_box"] = entry["box"]
    record.setdefault("stats", [])
    record.setdefault("designations", [])
    record.setdefault("named_features", [])
    record.setdefault("notable_facts", [])
    record.setdefault("cross_border", {"is_transboundary": False,
                                       "partner_countries": [], "detail": None})
    return record


# --------------------------------------------------------- C. VALIDATE

def implausible(park: dict) -> list[str]:
    """Name every figure that can't be right, so it can be re-extracted."""
    problems = []
    checks = (
        ("area_sq_km", PLAUSIBLE_AREA_SQ_KM),
        ("highest_point_m", PLAUSIBLE_HEIGHT_M),
        ("visitors_per_year", PLAUSIBLE_VISITORS),
    )
    for field, (low, high) in checks:
        value = park.get(field)
        if isinstance(value, (int, float)) and not (low <= value <= high):
            problems.append(f"{field}={value:,.0f} outside {low:,}-{high:,}")
    return problems


def recheck_park(record: dict, park: dict) -> dict:
    """Re-extract one park's stats from the box alone, and keep the fix only if
    it is actually more plausible than what we had."""
    previous = json.dumps(record.get("stats", []), ensure_ascii=False)
    prompt = RECHECK_PROMPT.format(previous=previous, box=record.get("raw_box", ""))
    try:
        fixed = call_llm_json(prompt, MODEL_EXTRACT,
                              label=f"recheck park {record['index']}")
    except LLMJSONError:
        return record

    candidate = dict(record, stats=fixed.get("stats") or record.get("stats", []))
    if implausible(normalize_park(candidate)):
        return record  # the retry is no better; keep the original and flag it
    return candidate


# ------------------------------------------------------ D. EXTRA AGGREGATES

def extend_aggregates(parks: list[dict], records: list[dict]) -> dict:
    """
    Aggregates over the prose fields, computed in Python for the same reason as
    the numeric ones: the model reads these off, it does not assemble them.
    """
    by_index = {r.get("index"): r for r in records}

    transboundary, designations, features = [], {}, {}
    for park in parks:
        record = by_index.get(park.get("index"), {})

        border = record.get("cross_border") or {}
        if border.get("is_transboundary"):
            transboundary.append({
                "name": park.get("name"),
                "country": park.get("country"),
                "partner_countries": border.get("partner_countries") or [],
                "detail": border.get("detail"),
            })

        for designation in record.get("designations") or []:
            designations.setdefault(str(designation).strip(), []).append(park.get("name"))

        for feature in record.get("named_features") or []:
            features.setdefault(str(feature).strip(), []).append(park.get("name"))

    return {
        "transboundary_or_paired_parks": transboundary,
        "count_transboundary_or_paired": len(transboundary),
        "parks_by_designation": {d: sorted(set(n for n in names if n))
                                 for d, names in sorted(designations.items())},
        "named_features_index": sorted(features),
    }


# --------------------------------------------------------- BUILD THE STORE

def build_store(document_text: str, narrative_tree_path: str = "tree.json",
                max_workers: int = 6, checkpoint_path: str = "store_v4.json",
                resume: bool = True) -> dict:
    """Extract every park entry, validate the figures, and compute everything."""
    entries = find_park_entries(document_text)
    print(f"Split document into {len(entries)} park entries "
          f"(median {sum(len(e['entry'].split()) for e in entries) // len(entries)} words each).")
    if not entries:
        raise RuntimeError(
            f"No '{BOX_ANCHOR}' boxes found. This pipeline is anchored on that "
            "phrase -- use pipeline_v2.py for a document without it."
        )

    done: dict[int, dict] = {}
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for record in json.load(f).get("raw_records", []):
                    if isinstance(record.get("index"), int):
                        done[record["index"]] = record
            if done:
                print(f"Resuming: {len(done)}/{len(entries)} entries already extracted.")
        except (json.JSONDecodeError, OSError):
            print("  ! store checkpoint unreadable; extracting from scratch")

    todo = [e for e in entries if e["index"] not in done]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_park, e): e["index"] for e in todo}
        for future in as_completed(futures):
            record = future.result()
            done[record["index"]] = record
            print(f"[extract] {len(done)}/{len(entries)} "
                  f"({record.get('name') or 'unnamed'})")

    records = [done[i] for i in sorted(done)]
    parks = [normalize_park(r) for r in records]

    # Validation pass -- only the parks that actually look wrong get re-read.
    suspect = [(r, p) for r, p in zip(records, parks) if implausible(p)]
    if suspect:
        print(f"\n[validate] {len(suspect)} parks have implausible figures; re-reading:")
        for record, park in suspect:
            print(f"  - {park.get('name')}: {'; '.join(implausible(park))}")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fixed = list(pool.map(lambda rp: recheck_park(*rp), suspect))
        for record in fixed:
            done[record["index"]] = record
        records = [done[i] for i in sorted(done)]
        parks = [normalize_park(r) for r in records]

    for park in parks:
        problems = implausible(park)
        if problems:
            # Still wrong after a re-read: mark it so the answer model knows
            # not to trust this row, rather than silently ranking on it.
            park["figures_unreliable"] = problems
            print(f"  ! {park.get('name')} still implausible: {'; '.join(problems)}")

    computed = compute_aggregates(parks)
    computed.update(extend_aggregates(parks, records))

    # Carry the prose fields onto each row so needle and cross-section
    # questions have something to read.
    by_index = {r.get("index"): r for r in records}
    for park in parks:
        record = by_index.get(park.get("index"), {})
        park["designations"] = record.get("designations") or []
        park["cross_border"] = record.get("cross_border") or {}
        park["named_features"] = record.get("named_features") or []
        park["notable_facts"] = record.get("notable_facts") or []
        park["country_evidence"] = record.get("country_evidence")
        park.pop("raw_box", None)  # the stats already carry it, and it is bulky

    narratives = load_narratives(narrative_tree_path)

    store = {
        "parks": parks,
        "computed": computed,
        "narratives": narratives,
        "raw_records": records,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    return store


def load_narratives(path: str) -> dict:
    """Reuse the ordered summaries pipeline_v2 already paid for."""
    if not os.path.exists(path):
        print(f"  ! {path} not found -- global-synthesis questions will be weaker.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"  ! {path} unreadable; continuing without narratives")
        return {}

    topics = []
    for node in tree.get("level1", []):
        for topic in node.get("topics_mentioned") or []:
            if topic not in topics:
                topics.append(topic)

    print(f"Reused {len(tree.get('level2', []))} section summaries and "
          f"{len(topics)} topics from {path}.")
    return {
        "document_arc": tree.get("level3", {}),
        "sections": [
            {"position_range": n.get("position_range"),
             "summary": n.get("narrative_summary")}
            for n in tree.get("level2", [])
        ],
        "topic_index": sorted(topics),
    }


# ------------------------------------------------------------- ANSWER

ANSWER_PROMPT = """Answer the question using ONLY the data below, compiled from a book of
European national parks. Do not use outside knowledge.

How to use each part:
- "computed" was calculated in code from every park's figures and prose. For
  any counting, ranking, "how many" or "which is the largest/highest/most"
  question, TRUST COMPUTED OVER YOUR OWN COUNTING. Do not recount by hand.
- "parks" is the complete table, one row per park: normalized figures (areas
  in sq km, heights in metres) plus that park's designations, cross-border
  status, named features and notable facts. Use notable_facts for specific
  details, dates and firsts.
- A row carrying "figures_unreliable" had a figure that could not be read from
  the scrambled source box. Do not rank or count on that row's numbers.
- "narratives" holds the ordered section summaries and the document's arc --
  use these for themes and for how the book treats a subject across entries.
- "topic_index" lists every topic the book mentions. For absence questions, a
  subject missing from the topic index, the designations and the notable facts
  is genuinely absent; say so explicitly.

- Commit to a single confident answer. Do not hedge or list alternatives --
  a wrong committed answer scores better than a hedge, and hedging can itself
  trigger a forbidden claim.
- If the data lacks enough information, still give your best single answer
  rather than refusing -- an unanswered question scores zero either way.
- When the question asks you to name things, name them all, not a sample.
- When a question asks for a figure the book prints, give the book's own units
  as well as the normalized value if they differ.

DATA (JSON):
{store}

QUESTION:
{question}

Answer in 1-4 sentences, plain text, no JSON, no preamble.
"""


def answer_context(store: dict) -> dict:
    """Everything except the raw extraction records. No question-dependent
    selection: every question sees exactly the same data."""
    return {k: v for k, v in store.items() if k != "raw_records"}


def answer_question(store: dict, question: str) -> str:
    context = json.dumps(answer_context(store), separators=(",", ":"),
                         ensure_ascii=False)
    prompt = ANSWER_PROMPT.format(store=context, question=question)
    return call_llm(prompt, model=MODEL_ANSWER).strip()


def run_submission(store: dict, questions: list[dict], output_path: str,
                   team: str = "group_c", notes: str = "") -> None:
    """Answer every question, writing the submission JSON incrementally."""
    submission = {"team": team, "notes": notes, "answers": []}

    for q in questions:
        qid = q["id"]
        print(f"Answering {qid}...")
        entry = {"id": qid}
        try:
            entry["answer"] = answer_question(store, q["question"])
        except Exception as e:
            entry["answer"] = ""
            entry["error"] = str(e)
            print(f"  -> failed: {e}")

        submission["answers"].append(entry)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Submission written to {output_path} ({len(submission['answers'])} answers)")


if __name__ == "__main__":
    with open("nationalparks_europe.txt", "r", encoding="utf-8") as f:
        document_text = preprocess_document(f.read())

    store = build_store(document_text, narrative_tree_path="tree.json")

    computed = store["computed"]
    context_chars = len(json.dumps(answer_context(store), separators=(",", ":")))
    print(f"\nAnswer context ~= {context_chars // 3:,} tokens")
    print(f"  parks: {computed['total_parks_profiled']}")
    print(f"  countries: {len(computed['parks_by_country'])}")
    print(f"  missing country: {computed['parks_missing_country']}")
    print(f"  missing area: {computed['parks_missing_area']}")
    print(f"  largest by area: {computed['largest_by_area'][:3]}")
    print(f"  transboundary: {computed['count_transboundary_or_paired']}")

    questions = load_questions("dev_questions.yaml")
    run_submission(
        store,
        questions,
        output_path="submission_v4.json",
        team="group_c",
        notes="Full park entries (heading through itineraries) extracted per "
              "'Park in numbers' anchor; figures validated against plausibility "
              "ranges and re-read when out of range; all counting, ranking and "
              "indexing computed in Python; narrative levels reused from the v2 tree.",
    )
