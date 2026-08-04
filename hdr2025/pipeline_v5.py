"""
Model E: contradiction engine + verified number/label pairing.

v4 scored 55.8% and fixed the prose categories (Needle 100%, Cross-section 67%),
but contradiction COLLAPSED:

    Contradiction:  v2 16.7%  ->  v3 38.9%  ->  v4 2.8%

Two separate causes, and neither is a prompting problem.

  1. NOTHING EVER COMPUTED A CONTRADICTION. The only conflict any version has
     detected is the area-unit outlier. d11 -- "the book calls one park the
     largest in its country but gives another a larger area" -- has scored 0%
     in every version, because finding it requires comparing a PROSE CLAIM
     against the TABLE, and no version extracted prose claims at all.
  2. v3 scored 67% on d12 by surfacing area_unit_outliers; v4 computes the
     identical field but buries it under 60 parks' worth of notable_facts and
     a several-hundred-entry feature index. Same signal, more noise, 8%.

Separately, two diagnosed figure bugs:

  d06 (0%): only two boxes report visitors -- "800,000 Approximate number of
      visitors annually" and "15 million visitors per year". The multiplier was
      read from the label only, so 15 million became 15 and 800,000 won.
      Fixed in pipeline_v3.normalize_park.
  d02 (15%): box 10 is "Area covered (sq km) 4528 1309 Highest point: Ben
      Macdui (m)". Paired backwards that is a 4528m summit, which passes the
      plausibility range -- a SWAP is invisible to a range check because both
      values are individually sane.

So v5 adds:

  A. TOKENIZED BOXES. Numbers and labels strictly alternate in these boxes and
     a number always sits adjacent to its own label. The extractor now sees the
     box as an explicit NUM/TEXT sequence and must use each number at most
     once; Python verifies the result is a real bijection over the box's
     numbers rather than trusting it.
  B. SUPERLATIVE CLAIMS as a first-class extracted field, then a Python
     contradiction engine that cross-checks every claim against the computed
     table and emits the conflicts it finds.
  C. A TRIMMED answer context, so the computed findings are not buried.

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
from pipeline_v3 import normalize_park, compute_aggregates
from pipeline_v4 import (
    find_park_entries,
    implausible,
    load_narratives,
    extend_aggregates,
    recheck_park,
)

MODEL_EXTRACT = "mistral-small-3-2"
MODEL_ANSWER = "mistral-small-3-2"

_NUM_TOKEN = r"\d[\d,]*(?:\.\d+)?(?:\s*[-/]\s*\d[\d,]*(?:\.\d+)?)?"


# ------------------------------------------------------- A. TOKENIZE A BOX

def tokenize_box(box: str) -> tuple[list[dict], list[str]]:
    """
    Split a scrambled box into its alternating NUMBER and LABEL tokens.

    The PDF conversion separated numbers from labels but preserved their order,
    so a number is always adjacent to the label it belongs to -- sometimes
    before it, sometimes after. Showing the model that sequence explicitly is a
    much stronger hint than showing it the flattened text, and it lets us check
    afterwards that every number was used at most once.
    """
    flat = re.sub(r"<!--.*?-->", " ", box)
    flat = re.sub(r"^\s*Park in numbers", "", " ".join(flat.split()), flags=re.I)

    tokens, numbers = [], []
    for piece in re.split(f"({_NUM_TOKEN})", flat):
        piece = (piece or "").strip()
        if not piece:
            continue
        if re.fullmatch(_NUM_TOKEN, piece):
            tokens.append({"kind": "NUMBER", "text": piece})
            numbers.append(piece)
        else:
            tokens.append({"kind": "LABEL", "text": piece})
    return tokens, numbers


def render_tokens(tokens: list[dict]) -> str:
    return "\n".join(f"  {i}. {t['kind']:<6} {t['text']}"
                     for i, t in enumerate(tokens, start=1))


def _canonical_number(raw: str) -> str:
    return re.sub(r"[,\s]", "", str(raw or ""))


def stats_disagree_with_box(stats: list[dict], numbers: list[str]) -> list[str]:
    """
    Check the model's pairing is a valid assignment over the box's own numbers.

    Catches invented values and the same number claimed by two stats. It cannot
    catch a clean swap of two numbers -- that is what the token sequence in the
    prompt is for.
    """
    available = [_canonical_number(n) for n in numbers]
    problems, used = [], []

    for stat in stats:
        value = _canonical_number(stat.get("value"))
        if not value:
            continue
        if value not in available:
            problems.append(f"{stat.get('label')!r} = {stat.get('value')!r} "
                            f"is not a number printed in the box")
        elif value in used:
            problems.append(f"number {stat.get('value')!r} assigned to more "
                            f"than one label")
        else:
            used.append(value)
    return problems


# ------------------------------------------------------------ B. EXTRACT

EXTRACT_PROMPT = """You are reading one complete park entry from a book of European national parks.

Extract ONLY what this text states. Never infer from general knowledge -- if
the text does not say it, use null or an empty list.

## Pairing the numbers box

PDF conversion separated each number from its label but KEPT THEIR ORDER, so
every number sits directly next to the label it belongs to -- sometimes just
before it, sometimes just after. The box is given to you below as a numbered
token sequence.

Rules, in order of priority:
1. Pair each label with the NEAREST number that no other label has taken.
2. Use each number at most once. Never invent a number that is not in the list.
3. Sanity check: a park's area is 1-15,000 sq km and a European summit is under
   5,000 m. If a pairing breaks that, you took a number belonging to another
   stat (an age, a species count, a wildlife population).

Worked example. For the sequence
    1. LABEL  Area covered (sq km)
    2. NUMBER 4528
    3. NUMBER 1309
    4. LABEL  Highest point: Ben Macdui (m)
the answer is area = 4528 (the number after its label) and highest point = 1309
(the number before its label). Reading it as a 4528m summit is wrong.

## Country

Quote the exact sentence you took the country from in "country_evidence".
Entries often name neighbouring countries for travel directions ("accessible
via Croatia"). The country is where the PARK is, not where you travel from.

## Superlative claims

Record every claim the PROSE makes that something is the largest, highest,
longest, oldest, deepest, first or most-visited of its kind, with the scope it
claims ("in Wales", "in Britain", "in Spain", "in Europe"). Copy the claim
sentence verbatim. These are checked against the book's own figures later, so
record them even when they look obviously true.

Return valid JSON, no prose, matching exactly this schema:

{{
  "name": "the park's name as the book writes it",
  "country": "country the park is in, or null if not stated",
  "country_evidence": "the exact sentence stating the country, or null",
  "stats": [
    {{"label": "the LABEL token, exactly as printed",
      "value": "the NUMBER token you paired with it, exactly as printed"}}
  ],
  "superlative_claims": [
    {{"text": "the claim sentence, verbatim",
      "subject": "what the claim is about, e.g. 'Snowdon' or this park's name",
      "quality": "largest | highest | longest | oldest | deepest | most-visited | first | other",
      "scope": "what it claims to be the -est of, e.g. 'Wales', 'Britain', 'Spain', 'Europe'",
      "value": "the figure given for it, or null"}}
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
    "include firsts and foundings, wildlife, conservation status and threats"
  ]
}}

PARK IN NUMBERS BOX, as a token sequence:
{tokens}

FULL ENTRY TEXT:
{entry}
"""


def extract_park(entry: dict) -> dict:
    """Extract one park's full record. Never raises."""
    tokens, numbers = tokenize_box(entry["box"])
    prompt = EXTRACT_PROMPT.format(
        tokens=render_tokens(tokens), entry=entry["entry"]
    )
    try:
        record = call_llm_json(prompt, MODEL_EXTRACT,
                               label=f"park {entry['index']}")
    except LLMJSONError as e:
        print(f"  -> park {entry['index']} extraction failed: {e}")
        record = {}

    record["index"] = entry["index"]
    record["raw_box"] = entry["box"]
    record["box_numbers"] = numbers
    for field, default in (("stats", []), ("superlative_claims", []),
                           ("designations", []), ("named_features", []),
                           ("notable_facts", [])):
        record.setdefault(field, default)
    record.setdefault("cross_border", {"is_transboundary": False,
                                       "partner_countries": [], "detail": None})
    return record


# ------------------------------------------------ C. CONTRADICTION ENGINE

# Scopes a claim can use that span more than one of the table's countries.
SCOPE_GROUPS = {
    "britain": {"england", "scotland", "wales", "united kingdom", "uk",
                "great britain", "britain"},
    "great britain": {"england", "scotland", "wales", "united kingdom", "uk"},
    "the uk": {"england", "scotland", "wales", "northern ireland",
               "united kingdom", "uk"},
    "scandinavia": {"norway", "sweden", "denmark"},
    "iberia": {"spain", "portugal"},
}


def _norm_name(value: str | None) -> str:
    """Fold park names for comparison: 'Ordesa National Park' -> 'ordesa'."""
    text = str(value or "").lower()
    text = re.sub(r"\b(national|nature|natural|regional)\b", " ", text)
    text = re.sub(r"\bparks?\b", " ", text)
    return " ".join(text.split())


def _names_match(a: str | None, b: str | None) -> bool:
    left, right = _norm_name(a), _norm_name(b)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _scope_countries(scope: str | None, known: set[str]) -> set[str]:
    """Which of the table's countries a claim's scope covers."""
    key = str(scope or "").strip().lower().removeprefix("the ").strip()
    if not key:
        return set()
    if key in SCOPE_GROUPS:
        return {c for c in known if c.lower() in SCOPE_GROUPS[key]}
    return {c for c in known if c.lower() == key}


def find_contradictions(parks: list[dict], records: list[dict],
                        computed: dict) -> list[dict]:
    """
    Cross-check every prose claim against the computed figures.

    This is the piece every earlier version was missing. d11 asks for exactly
    this comparison -- a "largest in its country" claim versus the book's own
    area column -- and no amount of prompting gets there when the claims were
    never extracted and the comparison was never made.
    """
    findings = []
    countries = {(p.get("country") or "").strip()
                 for p in parks if p.get("country")}

    # 1. Unit inconsistency. v3 scored 67% on this by surfacing it; make it a
    #    stated finding rather than a field the model has to notice.
    majority = computed.get("area_unit_majority")
    for unit, names in (computed.get("area_unit_outliers") or {}).items():
        findings.append({
            "type": "unit_inconsistency",
            "finding": f"{', '.join(n for n in names if n)} reports its area in "
                       f"'{unit}' while every other park uses '{majority}'.",
            "parks": names,
            "odd_unit": unit,
            "majority_unit": majority,
        })

    for record in records:
        park = next((p for p in parks if p.get("index") == record.get("index")), {})
        for claim in record.get("superlative_claims") or []:
            quality = str(claim.get("quality") or "").lower()
            scope = claim.get("scope")
            subject = claim.get("subject")
            in_scope = _scope_countries(scope, countries)
            if not in_scope:
                continue

            # 2. "largest park in <country>" vs the area column.
            if quality in ("largest", "biggest"):
                rivals = [
                    p for p in parks
                    if (p.get("country") or "").strip() in in_scope
                    and isinstance(p.get("area_sq_km"), (int, float))
                    and not p.get("figures_unreliable")
                ]
                if not rivals:
                    continue
                biggest = max(rivals, key=lambda p: p["area_sq_km"])
                claimed = next((p for p in rivals
                                if _names_match(p.get("name"), subject)), None)
                if not _names_match(biggest.get("name"), subject):
                    findings.append({
                        "type": "superlative_contradicted_by_figures",
                        "finding": f"The book calls {subject} the largest in "
                                   f"{scope}, but its own areas make "
                                   f"{biggest.get('name')} larger "
                                   f"({biggest['area_sq_km']:,.0f} sq km"
                                   + (f" vs {claimed['area_sq_km']:,.0f} sq km"
                                      if claimed else "") + ").",
                        "claim": claim.get("text"),
                        "claimed_subject": subject,
                        "larger_park": biggest.get("name"),
                        "larger_area_sq_km": biggest["area_sq_km"],
                        "claimed_area_sq_km": claimed.get("area_sq_km") if claimed else None,
                    })

            # 3. "highest in <scope>" vs the highest-point column.
            if quality == "highest":
                summits = [
                    p for p in parks
                    if (p.get("country") or "").strip() in in_scope
                    and isinstance(p.get("highest_point_m"), (int, float))
                    and not p.get("figures_unreliable")
                ]
                if not summits:
                    continue
                tallest = max(summits, key=lambda p: p["highest_point_m"])
                claimed_height = None
                for candidate in (claim.get("value"), park.get("highest_point_m")):
                    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(candidate or ""))
                    if match:
                        claimed_height = float(match.group().replace(",", ""))
                        break
                if (claimed_height is not None
                        and claimed_height < tallest["highest_point_m"]
                        and not _names_match(tallest.get("highest_point_name"), subject)):
                    findings.append({
                        "type": "superlative_contradicted_by_figures",
                        "finding": f"The book calls {subject} the highest in "
                                   f"{scope} at {claimed_height:,.0f}m, but its own "
                                   f"figures give {tallest.get('highest_point_name') or tallest.get('name')} "
                                   f"as {tallest['highest_point_m']:,.0f}m "
                                   f"(in {tallest.get('name')}).",
                        "claim": claim.get("text"),
                        "claimed_subject": subject,
                        "claimed_height_m": claimed_height,
                        "taller_summit": tallest.get("highest_point_name"),
                        "taller_height_m": tallest["highest_point_m"],
                        "taller_park": tallest.get("name"),
                    })

    return findings


# --------------------------------------------------------- BUILD THE STORE

def build_store(document_text: str, narrative_tree_path: str = "tree.json",
                max_workers: int = 6, checkpoint_path: str = "store_v5.json",
                resume: bool = True) -> dict:
    entries = find_park_entries(document_text)
    print(f"Split document into {len(entries)} park entries.")
    if not entries:
        raise RuntimeError("No 'Park in numbers' boxes found -- use pipeline_v2.py.")

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

    # Two independent checks: is the figure plausible, and is the pairing a
    # real assignment over the numbers actually printed in the box?
    suspect = []
    for record, park in zip(records, parks):
        problems = implausible(park) + stats_disagree_with_box(
            record.get("stats", []), record.get("box_numbers", [])
        )
        if problems:
            suspect.append((record, park, problems))

    if suspect:
        print(f"\n[validate] {len(suspect)} parks failed a check; re-reading:")
        for _, park, problems in suspect:
            print(f"  - {park.get('name')}: {'; '.join(problems)}")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fixed = list(pool.map(lambda s: recheck_park(s[0], s[1]), suspect))
        for record in fixed:
            done[record["index"]] = record
        records = [done[i] for i in sorted(done)]
        parks = [normalize_park(r) for r in records]

    for record, park in zip(records, parks):
        problems = implausible(park) + stats_disagree_with_box(
            record.get("stats", []), record.get("box_numbers", [])
        )
        if problems:
            park["figures_unreliable"] = problems
            print(f"  ! {park.get('name')} still failing: {'; '.join(problems)}")

    computed = compute_aggregates(parks)
    computed.update(extend_aggregates(parks, records))

    by_index = {r.get("index"): r for r in records}
    for park in parks:
        record = by_index.get(park.get("index"), {})
        park["designations"] = record.get("designations") or []
        park["cross_border"] = record.get("cross_border") or {}
        park["named_features"] = record.get("named_features") or []
        park["notable_facts"] = record.get("notable_facts") or []
        park["superlative_claims"] = record.get("superlative_claims") or []
        park["country_evidence"] = record.get("country_evidence")
        park.pop("raw_box", None)

    computed["contradictions"] = find_contradictions(parks, records, computed)
    computed["superlative_claims_index"] = [
        {"park": p.get("name"), **c}
        for p in parks for c in p.get("superlative_claims", [])
    ]
    # A several-hundred-entry list of every named feature crowded out the
    # computed findings in v4. The per-park named_features still carry it.
    computed.pop("named_features_index", None)

    store = {
        "parks": parks,
        "computed": computed,
        "narratives": load_narratives(narrative_tree_path),
        "raw_records": records,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    return store


# ------------------------------------------------------------- ANSWER

ANSWER_PROMPT = """Answer the question using ONLY the data below, compiled from a book of
European national parks. Do not use outside knowledge.

How to use each part:
- "computed" was calculated in code from every park's figures and prose. For
  any counting, ranking, "how many" or "which is the largest/highest/most"
  question, TRUST COMPUTED OVER YOUR OWN COUNTING. Do not recount by hand.
- "computed.contradictions" lists conflicts ALREADY VERIFIED IN CODE between
  what the book claims in prose and what its own figures say. For any question
  about the book contradicting itself, being inconsistent, or a claim its
  figures undercut, ANSWER FROM THIS LIST FIRST. Each entry states the claim,
  the figures that contradict it, and the parks involved. If the question
  matches one, report that finding directly and specifically, naming both
  sides and their numbers.
- "computed.superlative_claims_index" holds every "-est" claim the book makes,
  for conflicts the code could not adjudicate.
- "parks" is the complete table, one row per park: normalized figures (areas
  in sq km, heights in metres) plus designations, cross-border status, named
  features and notable facts. Use notable_facts for specific details and dates.
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
    return {k: v for k, v in store.items() if k != "raw_records"}


def answer_question(store: dict, question: str) -> str:
    context = json.dumps(answer_context(store), separators=(",", ":"),
                         ensure_ascii=False)
    prompt = ANSWER_PROMPT.format(store=context, question=question)
    return call_llm(prompt, model=MODEL_ANSWER).strip()


def run_submission(store: dict, questions: list[dict], output_path: str,
                   team: str = "group_c", notes: str = "") -> None:
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

    print(f"\nAnswer context ~= "
          f"{len(json.dumps(answer_context(store), separators=(',', ':'))) // 3:,} tokens")
    print(f"  parks: {computed['total_parks_profiled']}, "
          f"countries: {len(computed['parks_by_country'])}")
    print(f"  missing country: {computed['parks_missing_country']}")
    print(f"  largest by area: {computed['largest_by_area'][:2]}")
    print(f"  most visitors:   {computed['most_visitors'][:2]}")
    print(f"  summits >=3000m: {computed['count_highest_point_3000m_or_more']}")
    print(f"  superlative claims found: {len(computed['superlative_claims_index'])}")
    print(f"\n  CONTRADICTIONS FOUND: {len(computed['contradictions'])}")
    for finding in computed["contradictions"]:
        print(f"    [{finding['type']}] {finding['finding']}")

    questions = load_questions("dev_questions.yaml")
    run_submission(
        store, questions, output_path="submission_v5.json", team="group_c",
        notes="Park entries extracted with tokenized number/label pairing and "
              "bijection validation; superlative claims cross-checked against "
              "the computed figures to derive contradictions in Python; all "
              "counting and ranking computed, not generated.",
    )
