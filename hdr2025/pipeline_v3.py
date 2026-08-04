"""
Model C: structured park table + deterministic aggregates.

Why a third model. Scoring pipeline_v2 on the dev set gave 38.6% overall, but
the breakdown is what matters:

    Needle 83% | Superlative 56% | Global synthesis 53%
    Cross-section 33% | Absence 28% | Contradiction 17% | Aggregation 0%

Aggregation at 0/3 is not a tuning problem. "How many parks are in Spain",
"how many report a highest point over 3000m" -- those are computations over a
complete enumeration. v2 asked an LLM to count from prose that had already been
summarized twice and then dropped level 1 from the answer context entirely.
Counting from lossy summaries cannot be fixed with a better prompt.

This document is a book of 60 park profiles, each with a "Park in numbers"
box. That is a table. So:

  A. PYTHON finds all 60 boxes by regex (exact, free, no recall risk).
  B. THE LLM untangles one box each -- the PDF conversion scrambled the
     number/label order, which is the one thing a regex can't do and a cheap
     model does easily. 60 small calls.
  C. PYTHON normalizes the values (sq miles -> sq km, ft -> m) and computes
     every aggregate, superlative and unit outlier deterministically.
  D. The narrative levels from pipeline_v2's tree.json are reused for the
     global-synthesis questions, which the summary tree already handled well.

The answer prompt gets the whole compact store -- the park table, the computed
aggregates, the topic index and the narratives -- for every question. It is
~30k tokens, so nothing is ever dropped to fit, and no question sees a
narrower view of the document than any other. Still no RAG.

Requires:
    pip install openai pyyaml --break-system-packages
    export BROKER_API_KEY=your_group_key
"""

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline_ed2 import (
    call_llm,
    call_llm_json,
    LLMJSONError,
    preprocess_document,
    load_questions,
)

MODEL_EXTRACT = "ministral-3b-2512"   # 60 small, highly constrained calls
MODEL_ANSWER = "mistral-small-3-2"    # graded output, once per question

BOX_ANCHOR = "Park in numbers"


# ------------------------------------------------------- A. FIND THE BOXES

def find_park_sections(text: str, context_words: int = 1200) -> list[dict]:
    """
    Locate every "Park in numbers" box and the text that leads up to it.

    Anchoring on the box rather than on headings is deliberate: the PDF
    conversion dropped some park headings entirely (Abruzzo has none), but all
    60 boxes survived.

    The context is the text immediately BEFORE the box, not after it. Each
    entry runs heading -> intro -> Toolbox -> Getting there -> box, so the
    park's name and its country both sit just above the box. At 1200 words a
    country name is present for all 60 parks; at 300 it is present for only 51,
    which is where the country-based counting questions would start losing rows.
    """
    anchors = [m.start() for m in re.finditer(re.escape(BOX_ANCHOR), text)]
    sections = []
    prev_end = 0

    for i, start in enumerate(anchors, start=1):
        heading = text.find("## ", start)
        box_end = heading if heading != -1 else min(start + 900, len(text))
        box = text[start:box_end].strip()

        # Never reach back past the previous park's box -- that text belongs to
        # another park and would put the wrong country in front of the model.
        window = text[prev_end:start].split()
        context = " ".join(window[-context_words:])

        sections.append({"index": i, "context": context, "box": box})
        prev_end = box_end

    return sections


# ------------------------------------------------- B. UNTANGLE ONE BOX

EXTRACT_PROMPT = """You are reading one park entry from a book of European national parks.

The "Park in numbers" box was scrambled by PDF conversion: a number and the
label it belongs to are often separated, and sometimes the number comes after
its label instead of before. Pair each number with its correct label.

Extract ONLY what the text states. Do not infer values from general knowledge.
If the country is not stated in the text, use null -- do not guess it from the
park's name.

Return valid JSON, no prose, matching exactly this schema:

{{
  "name": "the park's name as the book writes it",
  "country": "country the park is in, or null if not stated in this text",
  "stats": [
    {{"label": "the label exactly as printed, e.g. 'Area covered (sq km)'",
      "value": "the number exactly as printed, e.g. '1,200' or '370-550'"}}
  ]
}}

The country is usually stated in the entry text, often in the "Getting there"
paragraph ("In central Italy, the park is accessible via...") or in the
opening description. Use that.

ENTRY TEXT (the passage leading up to the box):
{context}

PARK IN NUMBERS BOX:
{box}
"""


def extract_park(section: dict) -> dict:
    """Turn one scrambled box into a clean record. Never raises."""
    prompt = EXTRACT_PROMPT.format(context=section["context"], box=section["box"])
    try:
        record = call_llm_json(
            prompt, MODEL_EXTRACT, label=f"park {section['index']}"
        )
    except LLMJSONError as e:
        print(f"  -> park {section['index']} extraction failed: {e}")
        record = {"name": None, "country": None, "stats": []}

    record["index"] = section["index"]
    record["raw_box"] = section["box"]  # keep the source so nothing is unverifiable
    record.setdefault("stats", [])
    return record


# --------------------------------------------- C. NORMALIZE AND COMPUTE

# Canonical conversions into sq km and metres. Order matters: check "sq mile"
# before "mile" and "hectare" before "acre" is irrelevant, but "sq km" must be
# matched before a bare "km" would be.
_AREA_TO_SQKM = [
    (r"sq\.?\s*k(?:m|ilomet)", 1.0),
    (r"km\s*(?:2|²)", 1.0),
    (r"square\s+kilomet", 1.0),
    (r"sq\.?\s*mi|square\s+mile", 2.58999),
    (r"hectare|\bha\b", 0.01),
    (r"\bacre", 0.00404686),
]
_HEIGHT_TO_M = [
    (r"\bft\b|feet|foot", 0.3048),
    (r"\bm\b|metre|meter", 1.0),
]

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# A bare "peak" or "summit" is not enough. "Peak annual population of migrating
# cranes" was read as a 135,000m summit, which invented a park above 3000m and
# cost d02 its count; "Highest recorded wind speed... Cairn Gorm summit (km/h)"
# is the same trap with a different unit.
_HEIGHT_LABEL_RE = re.compile(
    r"highest\s+point|highest\s+peak|highest\s+summit|\bsummit\b|"
    r"elevation|altitude|\bhighest\b.*\((?:m|ft)\)", re.IGNORECASE)
_HEIGHT_EXCLUDE_RE = re.compile(
    r"population|species|visitor|number\s+of|percentage|wind|speed|"
    r"temperature|rainfall|depth|age\b|year", re.IGNORECASE)
_PARENTHETICAL_UNIT_RE = re.compile(r"\(([^)]*)\)")
_HEIGHT_UNIT_RE = re.compile(r"\b(m|metres?|meters?|ft|feet)\b", re.IGNORECASE)


def _is_height_label(label: str) -> bool:
    """
    True only for a label that really names a park's highest point.

    Three tests, because each one alone lets a different impostor through: the
    label must use height vocabulary, must not be about a population, a count
    or a wind speed, and -- if it states a unit at all -- that unit must be a
    unit of length.
    """
    if not _HEIGHT_LABEL_RE.search(label) or _HEIGHT_EXCLUDE_RE.search(label):
        return False
    unit = _PARENTHETICAL_UNIT_RE.search(label)
    if unit and unit.group(1).strip() and not _HEIGHT_UNIT_RE.search(unit.group(1)):
        return False
    return True


def parse_number(raw: str) -> float | None:
    """
    First number in a string, commas stripped.

    A range like "370-550" resolves to its low end rather than being discarded:
    a slightly conservative area still sorts correctly and still counts.
    """
    if raw is None:
        return None
    match = _NUMBER_RE.search(str(raw).replace("–", "-"))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def _convert(label: str, value: float,
             table: list[tuple[str, float]]) -> tuple[float, bool] | None:
    """
    Convert to the canonical unit. Returns (value, unit_was_printed) or None
    when the label names no unit this table recognises.
    """
    for pattern, factor in table:
        if re.search(pattern, label, re.IGNORECASE):
            return value * factor, True
    return None


def normalize_park(record: dict) -> dict:
    """
    Derive comparable numeric fields from the raw stats.

    Every derived field keeps the label it came from, so a wrong reading is
    traceable back to the printed box rather than silently trusted.
    """
    park = {
        "index": record.get("index"),
        "name": record.get("name"),
        "country": record.get("country"),
        "stats": record.get("stats", []),
        "raw_box": record.get("raw_box", ""),
    }

    for stat in park["stats"]:
        label = str(stat.get("label") or "")
        value = parse_number(stat.get("value"))
        if value is None:
            continue

        # Each box carries one piece of trivia alongside the two standard
        # figures, and that trivia often collides with these patterns:
        # "Years that the Sami have lived in the park AREA", "Percentage
        # COVERED by forest". So take the first stat that actually converts,
        # and never let a later one overwrite it.
        if "area_sq_km" not in park and re.search(r"\barea\b|covered", label, re.I):
            converted = _convert(label, value, _AREA_TO_SQKM)
            if converted is not None:
                park["area_sq_km"] = round(converted[0], 1)
                park["area_label"] = label
                # The unit as printed -- d12 asks which park breaks convention.
                unit = re.search(r"\(([^)]*)\)", label)
                park["area_unit_as_printed"] = unit.group(1) if unit else None
                continue

        if "highest_point_m" not in park and _is_height_label(label):
            converted = _convert(label, value, _HEIGHT_TO_M)
            # Heights are printed in metres throughout; an unlabelled one is
            # metres, and dropping it would silently shrink the 3000m count.
            metres = converted[0] if converted else value
            park["highest_point_m"] = round(metres, 1)
            park["highest_point_label"] = label
            park["highest_point_unit_assumed"] = converted is None
            # "Highest point: Mount Kebnekaise (m)" -> "Mount Kebnekaise"
            named = re.search(r"highest point[:\s]+(.+?)\s*(?:\(|$)", label, re.I)
            if named:
                park["highest_point_name"] = named.group(1).strip()
            continue

        if "visitors_per_year" not in park and re.search(r"visitor|tourist", label, re.I):
            # "million" can sit in either half: the book prints both
            # "15 million visitors per year" and "Visitors per year (million)".
            # Checking only the label read 15 million as 15, which let a park
            # with 800,000 win the ranking.
            scale_text = f"{label} {stat.get('value') or ''}"
            if re.search(r"billion|bn\b", scale_text, re.I):
                multiplier = 1_000_000_000
            elif re.search(r"million|\bm\b", scale_text, re.I):
                multiplier = 1_000_000
            else:
                multiplier = 1
            park["visitors_per_year"] = int(round(value * multiplier))
            park["visitors_label"] = label
            continue

    return park


def compute_aggregates(parks: list[dict]) -> dict:
    """
    Everything countable, counted in Python.

    This is the whole point of v3: an LLM asked to count 60 rows will drift,
    but it will read a precomputed number off a table correctly. All of this is
    derived once, identically for every question.
    """
    def top(field: str, n: int = 5) -> list[dict]:
        having = [p for p in parks if isinstance(p.get(field), (int, float))]
        having.sort(key=lambda p: p[field], reverse=True)
        return [
            {"name": p.get("name"), "country": p.get("country"), field: p[field]}
            for p in having[:n]
        ]

    by_country: dict[str, list[str]] = {}
    for park in parks:
        country = (park.get("country") or "unknown").strip()
        by_country.setdefault(country, []).append(park.get("name"))

    over_3000 = [
        {"name": p.get("name"), "highest_point_m": p["highest_point_m"]}
        for p in parks
        if isinstance(p.get("highest_point_m"), (int, float))
        and p["highest_point_m"] >= 3000
    ]

    # Which parks state their area in something other than the common unit
    units: dict[str, list[str]] = {}
    for park in parks:
        printed = park.get("area_unit_as_printed")
        if printed:
            units.setdefault(printed.strip().lower(), []).append(park.get("name"))
    majority = max(units, key=lambda u: len(units[u])) if units else None
    outliers = {u: names for u, names in units.items() if u != majority}

    # Largest park per country, so a "largest in <country>" claim in the prose
    # can be checked against the book's own figures.
    largest_by_country = {}
    for country, names in by_country.items():
        members = [
            p for p in parks
            if (p.get("country") or "unknown").strip() == country
            and isinstance(p.get("area_sq_km"), (int, float))
        ]
        if members:
            biggest = max(members, key=lambda p: p["area_sq_km"])
            largest_by_country[country] = {
                "name": biggest.get("name"), "area_sq_km": biggest["area_sq_km"]
            }

    return {
        "total_parks_profiled": len(parks),
        "parks_by_country": {c: sorted(n for n in names if n)
                             for c, names in sorted(by_country.items())},
        "park_count_by_country": {c: len(names)
                                  for c, names in sorted(by_country.items())},
        "parks_with_highest_point_3000m_or_more": sorted(
            over_3000, key=lambda p: p["highest_point_m"], reverse=True
        ),
        "count_highest_point_3000m_or_more": len(over_3000),
        "largest_by_area": top("area_sq_km"),
        "highest_summits": top("highest_point_m"),
        "most_visitors": top("visitors_per_year"),
        "area_unit_majority": majority,
        "area_unit_outliers": outliers,
        "largest_park_per_country_by_area": largest_by_country,
        "parks_missing_area": [p.get("name") for p in parks
                               if not isinstance(p.get("area_sq_km"), (int, float))],
        "parks_missing_country": [p.get("name") for p in parks
                                  if not p.get("country")],
    }


# ------------------------------------------------------- D. BUILD THE STORE

def build_store(document_text: str, narrative_tree_path: str = "tree.json",
                max_workers: int = 6, checkpoint_path: str = "store_v3.json",
                resume: bool = True) -> dict:
    """
    Extract every park box, normalize, compute aggregates, and fold in the
    narrative levels that pipeline_v2 already paid for.
    """
    sections = find_park_sections(document_text)
    print(f"Found {len(sections)} '{BOX_ANCHOR}' boxes.")
    if not sections:
        raise RuntimeError(
            f"No '{BOX_ANCHOR}' boxes found. This pipeline is anchored on that "
            "phrase, so a document without it yields an empty table -- use "
            "pipeline_v2.py, or change BOX_ANCHOR to whatever repeated block "
            "this document uses for its per-entry figures."
        )

    done: dict[int, dict] = {}
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for record in json.load(f).get("raw_records", []):
                    if isinstance(record.get("index"), int):
                        done[record["index"]] = record
            if done:
                print(f"Resuming: {len(done)}/{len(sections)} boxes already extracted.")
        except (json.JSONDecodeError, OSError):
            print("  ! store checkpoint unreadable; extracting from scratch")

    todo = [s for s in sections if s["index"] not in done]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_park, s): s["index"] for s in todo}
        for future in as_completed(futures):
            record = future.result()
            done[record["index"]] = record
            print(f"[extract] {len(done)}/{len(sections)} "
                  f"({record.get('name') or 'unnamed'})")

    raw_records = [done[i] for i in sorted(done)]
    parks = [normalize_park(r) for r in raw_records]

    # Narrative levels from pipeline_v2 -- already built, free to reuse.
    narratives: dict = {}
    if os.path.exists(narrative_tree_path):
        try:
            with open(narrative_tree_path, "r", encoding="utf-8") as f:
                tree = json.load(f)
            narratives = {
                "document_arc": tree.get("level3", {}),
                "sections": [
                    {"position_range": n.get("position_range"),
                     "summary": n.get("narrative_summary")}
                    for n in tree.get("level2", [])
                ],
            }
            topics = []
            for node in tree.get("level1", []):
                for topic in node.get("topics_mentioned") or []:
                    if topic not in topics:
                        topics.append(topic)
            narratives["topic_index"] = sorted(topics)
            print(f"Reused {len(narratives['sections'])} section summaries and "
                  f"{len(topics)} topics from {narrative_tree_path}.")
        except (json.JSONDecodeError, OSError):
            print(f"  ! {narrative_tree_path} unreadable; continuing without narratives")
    else:
        print(f"  ! {narrative_tree_path} not found -- global-synthesis and absence "
              f"questions will be weaker. Run pipeline_v2.py first to build it.")

    store = {
        "parks": parks,
        "computed": compute_aggregates(parks),
        "narratives": narratives,
        "raw_records": raw_records,  # checkpoint only, stripped before answering
    }

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    return store


# ------------------------------------------------------------- E. ANSWER

ANSWER_PROMPT = """Answer the question using ONLY the data below, compiled from a book of
European national park profiles. Do not use outside knowledge.

How to use each part:
- "computed" was calculated deterministically in code from every park's
  printed figures. For any counting, ranking or "how many" question, TRUST
  COMPUTED OVER YOUR OWN COUNTING. Do not recount the park list by hand.
- "parks" is the complete table, one row per park, with each park's figures
  normalized (areas in sq km, heights in metres) alongside the box as printed.
- "narratives" carries the document's section summaries in order and its
  overall arc -- use these for questions about themes, development, or how
  the book treats a subject across entries.
- "topic_index" lists every topic the book mentions. For absence questions,
  a subject missing from both the topic index and the park table is genuinely
  absent; say so explicitly.

- Commit to a single confident answer. Do not hedge or list alternatives --
  a wrong committed answer scores better than a hedge, and hedging can itself
  trigger a forbidden claim.
- If the data lacks enough information, still give your best single answer
  rather than refusing -- an unanswered question scores zero either way.
- When the question asks you to name things, name them all, not a sample.

DATA (JSON):
{store}

QUESTION:
{question}

Answer in 1-4 sentences, plain text, no JSON, no preamble.
"""


def answer_context(store: dict) -> dict:
    """
    The whole store minus the raw extraction records.

    No trimming and no question-dependent selection: the compact table is
    small enough that every question sees exactly the same data.
    """
    return {k: v for k, v in store.items() if k != "raw_records"}


def answer_question(store: dict, question: str) -> str:
    """Answer one question against the full store."""
    context = json.dumps(answer_context(store), separators=(",", ":"),
                         ensure_ascii=False)
    prompt = ANSWER_PROMPT.format(store=context, question=question)
    return call_llm(prompt, model=MODEL_ANSWER).strip()


def run_submission(store: dict, questions: list[dict], output_path: str,
                   team: str = "yourteam", notes: str = "") -> None:
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

    context_chars = len(json.dumps(answer_context(store), separators=(",", ":")))
    print(f"\nAnswer context ~= {context_chars // 3:,} tokens "
          f"({store['computed']['total_parks_profiled']} parks, "
          f"{len(store['computed']['parks_by_country'])} countries)")

    questions = load_questions("dev_questions.yaml")
    run_submission(
        store,
        questions,
        output_path="submission_v3.json",
        team="yourteam",
        notes="Structured park table extracted per 'Park in numbers' box, "
              "aggregates and superlatives computed deterministically in Python, "
              "narrative levels reused from the v2 summary tree.",
    )
