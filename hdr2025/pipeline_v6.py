"""
Model F: deterministic indexes over the raw text, on top of v5's park table.

v5 scored 61.0%. Every remaining failure is a missing index, not a weak model:

    d09 (0%)  Absence: nothing in the pipeline indexes THREATS at all. The
              topic index is still v2's -- vague strings a 3B model produced
              while summarizing 1500-word chunks, never rebuilt since.
    d08 (33%) Absence: designations are free text, so "Unesco World Heritage
              Site" and "UNESCO World Heritage" never group together.
    d15 (20%) Cross-section: "two entries, far apart, each record a climbing
              first with a date" is a sorted index of year-bearing facts. The
              facts were extracted; they were never indexed.
    d11 (0%)  Contradiction: v5's engine works -- it fired for d10 and d12 --
              but the "largest in its country" claim is never extracted, so
              there is nothing to cross-check.
    d03 (25%) Aggregation: country is a pure LLM judgement with no validation,
              and it swings between runs (75% in v4, 25% in v5).
    d17/d18   Global synthesis: reads 13 generic section summaries from the v2
              tree, drowned in a large store.

So v6 adds six indexes, five of them computed in Python with no LLM at all:

  1. COVERAGE INDEX   -- exact term frequencies over the whole document. A
                         subject absent from it is genuinely not discussed.
                         This is what Absence has needed all along.
  2. SUPERLATIVE SCAN -- regex over every entry for "-est ... in <scope>"
                         claims, merged with the model's own extraction, so
                         the contradiction engine stops depending on the model
                         volunteering a claim.
  3. DATED EVENTS     -- every fact carrying a year, sorted, with its park.
  4. DESIGNATIONS     -- canonical buckets instead of free text.
  5. COUNTRY RESOLVER -- the model's answer reconciled against the country
                         names that actually occur in the entry.
  6. THEME INDEX      -- the document's own recurring terms mapped to the
                         parks that discuss them, for synthesis questions.

NOT RETRIEVAL. Every index is built once at ingest from the whole document and
shipped complete to every question. No question narrows what the model sees;
each one gets byte-identical data. The trimming in build_answer_context is by
size only, in a fixed order, never by relevance to a question.

Reuses store_v5.json's extractions if present -- the schema is unchanged, so a
v6 run costs zero LLM calls when v5 has already run.

Requires:
    pip install openai pyyaml --break-system-packages
    export BROKER_API_KEY=your_group_key
"""

import os
import re
import json
import time
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline_ed2 import (
    call_llm,
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
from pipeline_v5 import (
    extract_park,
    stats_disagree_with_box,
    find_contradictions,
)

MODEL_ANSWER = "mistral-small-3-2"

# The answer model tops out at 128k. Leave headroom for prompt and output.
ANSWER_TOKEN_BUDGET = 95_000

# Greedy decoding. See answer_question for why this matters more than it looks.
# Override without editing this file:
#   ANSWER_TEMPERATURE=none python pipeline_v6.py   -> broker default (runs #19/#21)
#   ANSWER_TEMPERATURE=0.3 python pipeline_v6.py    -> sample a little
_TEMPERATURE_ENV = os.environ.get("ANSWER_TEMPERATURE", "0").strip().lower()
ANSWER_TEMPERATURE = (None if _TEMPERATURE_ENV in ("none", "default", "")
                      else float(_TEMPERATURE_ENV))


# ---------------------------------------------------------------- TIMING

STAGE_SECONDS: dict[str, float] = {}


def format_duration(seconds: float) -> str:
    """'1m 23.4s' for anything over a minute, '4.2s' below it."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:04.1f}s"


@contextmanager
def timed(label: str):
    """
    Time one stage, print it as it finishes, and accumulate it in
    STAGE_SECONDS so the run can report where the wall clock actually went.
    Re-entering the same label adds to it rather than overwriting.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        STAGE_SECONDS[label] = STAGE_SECONDS.get(label, 0.0) + elapsed
        print(f"  [{format_duration(elapsed):>9}] {label}")


# ------------------------------------------------------ 1. COVERAGE INDEX

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can cannot could did do does doing down during
each few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other others our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with
you your yours yourself yourselves it's don't isn't one two three four five six seven
eight nine ten first second third new old many much make made take taken get got go
goes going come comes came see seen look looks like well back even still way ways
around across along among within without near next best good great big small long short
high low top end ends part parts area areas place places time times year years day days
page marker start
""".split())

_WORD_RE = re.compile(r"[a-z][a-z'\-]{2,}")


def build_coverage_index(text: str, min_count: int = 3,
                         max_terms: int = 8000) -> dict:
    """
    Exact term frequencies over the entire document.

    This is the Absence fix. A vector index cannot answer "is poaching
    discussed?" because it always returns a nearest neighbour whether or not
    the topic exists. Counting the actual words can: a term that appears twice
    in 93,000 words is not substantively discussed, and one that appears zero
    times is not discussed at all.

    Unigrams and bigrams, stopword-filtered, with everything below min_count
    dropped -- the threshold itself carries the meaning, so omission from this
    index IS the evidence of absence.
    """
    flat = re.sub(r"<!--.*?-->", " ", text)
    flat = re.sub(r"[#*_>`\[\]()]", " ", flat).lower()
    words = _WORD_RE.findall(flat)

    counts = Counter(w for w in words if w not in STOPWORDS)
    for left, right in zip(words, words[1:]):
        if left not in STOPWORDS and right not in STOPWORDS:
            counts[f"{left} {right}"] += 1

    frequent = {term: n for term, n in counts.most_common(max_terms)
                if n >= min_count}
    return {
        "total_words": len(words),
        "min_count": min_count,
        # max_terms is a runaway guard, not a ranking cut: it sits far above the
        # count>=min_count population so it never truncates a qualifying term.
        # An earlier 2200 cap silently zeroed real terms, which would have made
        # the model call a discussed subject absent -- the exact error this
        # index exists to prevent.
        "note": (f"Exact counts over the whole document. Terms occurring fewer "
                 f"than {min_count} times are omitted, so a subject absent from "
                 f"this list is not substantively discussed."),
        "terms": dict(sorted(frequent.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def term_presence(index: dict, phrase: str) -> dict:
    """Look one subject up in the coverage index. Used for self-checks."""
    key = " ".join(_WORD_RE.findall(phrase.lower()))
    count = index.get("terms", {}).get(key, 0)
    return {"term": key, "count": count,
            "substantive": count >= index.get("min_count", 3)}


# -------------------------------------------------- 2. SUPERLATIVE SCANNER

QUALITY_WORDS = {
    "largest": "largest", "biggest": "largest", "greatest": "largest",
    "highest": "highest", "tallest": "highest", "loftiest": "highest",
    "longest": "longest", "oldest": "oldest", "deepest": "deepest",
    "smallest": "smallest", "first": "first",
    "most visited": "most-visited", "most popular": "most-visited",
}
_QUALITY_ALT = "|".join(sorted(QUALITY_WORDS, key=len, reverse=True))

# "the largest national park in Spain", "Europe's highest peak"
_SUPERLATIVE_IN = re.compile(
    rf"\b(?:the\s+)?({_QUALITY_ALT})\b[^.]{{0,70}}?\b(?:in|of)\s+"
    rf"(?:the\s+)?([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){{0,2}})",
    re.IGNORECASE,
)
_SUPERLATIVE_POSSESSIVE = re.compile(
    rf"\b([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){{0,1}})['’]s\s+"
    rf"(?:only\s+|last\s+)?({_QUALITY_ALT})\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def scan_entry_superlatives(entry_text: str, park_name: str | None) -> list[dict]:
    """
    Regex every "-est ... in <scope>" claim out of one park's entry.

    v5 relied on the model listing its own superlative claims, and d11 stayed
    at 0% because it never listed the one that mattered. A regex has no
    judgement but it has perfect recall on the phrasings it covers, which is
    the property the contradiction engine actually needs -- a claim the engine
    never sees can never be checked.
    """
    claims = []
    for sentence in split_sentences(entry_text):
        if len(sentence) > 400:
            continue
        for match in _SUPERLATIVE_IN.finditer(sentence):
            quality = QUALITY_WORDS.get(match.group(1).lower())
            scope = match.group(2).strip()
            if not quality or scope.lower() in ("the", "a"):
                continue
            claims.append({
                "text": sentence, "subject": park_name, "quality": quality,
                "scope": scope, "value": None, "source": "regex",
            })
        for match in _SUPERLATIVE_POSSESSIVE.finditer(sentence):
            quality = QUALITY_WORDS.get(match.group(2).lower())
            if not quality:
                continue
            claims.append({
                "text": sentence, "subject": park_name, "quality": quality,
                "scope": match.group(1).strip(), "value": None, "source": "regex",
            })
    return claims


def merge_claims(model_claims: list[dict], regex_claims: list[dict]) -> list[dict]:
    """Union the model's claims with the scanner's, deduped on text+quality."""
    merged, seen = [], set()
    for claim in list(model_claims) + list(regex_claims):
        key = (" ".join(str(claim.get("text") or "").lower().split())[:120],
               str(claim.get("quality") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        claim.setdefault("source", "model")
        merged.append(claim)
    return merged


# ---------------------------------------------------- 3. DATED EVENTS INDEX

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def build_dated_events(parks: list[dict]) -> list[dict]:
    """
    Every extracted fact that carries a year, sorted chronologically.

    d15 asks for two climbing firsts recorded in entries far apart in the book.
    The facts were already extracted in v5 -- they were simply scattered across
    60 rows with no way to line them up.
    """
    events = []
    for park in parks:
        for fact in park.get("notable_facts") or []:
            for year in _YEAR_RE.findall(str(fact)):
                events.append({
                    "year": int(year),
                    "park": park.get("name"),
                    "country": park.get("country"),
                    "fact": str(fact).strip(),
                })
                break  # one entry per fact, keyed on its first year
    events.sort(key=lambda e: (e["year"], e["park"] or ""))
    return events


def build_firsts_index(events: list[dict]) -> list[dict]:
    """Dated events that record a first/earliest -- climbs, ascents, foundings."""
    pattern = re.compile(r"\bfirst\b|\bearliest\b|\bmaiden\b|\binitial\s+ascent\b",
                         re.IGNORECASE)
    return [e for e in events if pattern.search(e["fact"])]


# ---------------------------------------------- 4. CANONICAL DESIGNATIONS

DESIGNATION_RULES = [
    (r"world\s+heritage", "Unesco World Heritage Site"),
    (r"biosphere", "Unesco Biosphere Reserve"),
    (r"geopark", "Unesco Global Geopark"),
    (r"natura\s*2000", "Natura 2000"),
    (r"ramsar", "Ramsar Wetland"),
    (r"dark\s+sky", "Dark Sky Reserve"),
    (r"national\s+scenic", "National Scenic Area"),
    (r"nature\s+reserve", "Nature Reserve"),
    (r"marine\s+(?:protected|park|reserve)", "Marine Protected Area"),
]


def canonical_designation(raw: str) -> str:
    """
    Fold designation spellings into one bucket each.

    "Unesco World Heritage Site", "UNESCO World Heritage", "a World Heritage
    listing" are one designation; as free text they were three, which is why
    d08 and d14 could never be counted correctly.
    """
    text = " ".join(str(raw or "").split())
    for pattern, canonical in DESIGNATION_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return canonical
    return text.strip() or "Unspecified"


# ------------------------------------------------------ 5. COUNTRY RESOLVER

EUROPEAN_COUNTRIES = [
    "Albania", "Andorra", "Austria", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "England",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland",
    "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania",
    "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
    "North Macedonia", "Northern Ireland", "Norway", "Poland", "Portugal",
    "Romania", "Scotland", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
    "Switzerland", "Turkey", "Ukraine", "Wales",
]
# Adjectives are often the only form used: "in Swedish Lappland".
COUNTRY_ADJECTIVES = {
    "Swedish": "Sweden", "Norwegian": "Norway", "Finnish": "Finland",
    "Danish": "Denmark", "Icelandic": "Iceland", "Italian": "Italy",
    "Spanish": "Spain", "Portuguese": "Portugal", "French": "France",
    "German": "Germany", "Austrian": "Austria", "Swiss": "Switzerland",
    "Polish": "Poland", "Czech": "Czech Republic", "Slovak": "Slovakia",
    "Slovenian": "Slovenia", "Croatian": "Croatia", "Serbian": "Serbia",
    "Montenegrin": "Montenegro", "Albanian": "Albania", "Greek": "Greece",
    "Romanian": "Romania", "Bulgarian": "Bulgaria", "Hungarian": "Hungary",
    "Estonian": "Estonia", "Latvian": "Latvia", "Lithuanian": "Lithuania",
    "Dutch": "Netherlands", "Welsh": "Wales", "Scottish": "Scotland",
    "English": "England", "Irish": "Ireland",
}


def country_mentions(entry_text: str) -> Counter:
    """Count every country named in one entry, directly or as an adjective."""
    counts = Counter()
    for country in EUROPEAN_COUNTRIES:
        n = len(re.findall(rf"\b{re.escape(country)}\b", entry_text))
        if n:
            counts[country] += n
    for adjective, country in COUNTRY_ADJECTIVES.items():
        n = len(re.findall(rf"\b{adjective}\b", entry_text))
        if n:
            counts[country] += n
    return counts


def resolve_country(model_country: str | None, entry_text: str) -> dict:
    """
    Reconcile the model's country against the entry's own text.

    Country attribution swung from 75% (v4) to 25% (v5) on the same question
    because nothing ever checked it. The entry text is the ground truth: a
    country the entry never names cannot be where the park is, and when the
    model's pick is not the dominant mention the row is marked uncertain
    rather than silently counted.
    """
    counts = country_mentions(entry_text)
    stated = (model_country or "").strip()
    normalized = next((c for c in EUROPEAN_COUNTRIES
                       if c.lower() == stated.lower()), stated)

    if not counts:
        return {"country": normalized or None, "country_source": "model_only",
                "country_uncertain": bool(normalized)}

    ranked = counts.most_common()
    dominant, dominant_n = ranked[0]
    runner_up_n = ranked[1][1] if len(ranked) > 1 else 0
    # "Clear" means the leader outweighs the next country two to one, which is
    # what separates the park's own country from one named only for directions.
    clear = dominant_n >= 2 * runner_up_n
    mentions = dict(ranked[:4])

    if not normalized:
        return {"country": dominant, "country_source": "text_scan",
                "country_uncertain": not clear, "country_mentions": mentions}

    if normalized not in counts:
        # The model named a country this entry never mentions -- it cannot be
        # where the park is, so the text wins outright.
        return {"country": dominant, "country_source": "text_override",
                "country_uncertain": not clear, "country_mentions": mentions}

    if normalized == dominant:
        return {"country": normalized, "country_source": "model_confirmed",
                "country_uncertain": False, "country_mentions": mentions}

    if counts[normalized] * 2 < dominant_n:
        # The model's pick is mentioned, but swamped -- almost always a
        # neighbour named for travel directions ("accessible via Croatia").
        return {"country": dominant, "country_source": "text_outweighed",
                "country_uncertain": not clear, "country_mentions": mentions}

    # Both plausible: keep the model's reading but say the row is not certain.
    return {"country": normalized, "country_source": "model_kept",
            "country_uncertain": True, "country_mentions": mentions}


# --------------------------------------------------------- 6. THEME INDEX

# Photo credits, the publisher's own name and the book's structural headings
# recur in nearly every entry without being themes of the document.
THEME_NOISE = re.compile(
    r"getty|alamy|shutterstock|images?\b|photo|lonely planet|"
    r"national park|nature reserve|in numbers|highest point|page|"
    r"visitor centre|getting there|where to stay|see also",
    re.IGNORECASE,
)


def build_theme_index(parks: list[dict], coverage: dict,
                      entry_by_index: dict[int, str],
                      max_themes: int = 30, max_parks: int = 14) -> dict:
    """
    Map the document's own recurring themes to the parks that discuss them.

    Deliberately data-driven rather than a hand-written theme list: the themes
    are the most frequent multi-word terms in the coverage index, so this works
    on any document instead of only this one. Synthesis questions then get
    cross-park evidence already assembled, rather than 13 generic section
    summaries.
    """
    candidates = [
        term for term in coverage.get("terms", {})
        if " " in term
        and not any(ch.isdigit() for ch in term)
        and not THEME_NOISE.search(term)
    ]

    # A term in almost every entry describes the book's format, not its themes.
    ubiquitous = max(4, int(0.85 * len(parks)))

    index = {}
    for term in candidates:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        hits = [
            park.get("name") for park in parks
            if pattern.search(entry_by_index.get(park.get("index"), ""))
        ]
        # A theme is only a theme if it recurs -- but not everywhere.
        if 4 <= len(hits) <= ubiquitous:
            index[term] = {
                "park_count": len(hits),
                "parks": [h for h in hits if h][:max_parks],
                "truncated": len(hits) > max_parks,
            }
        if len(index) >= max_themes:
            break
    return index


# --------------------------------------------------------- BUILD THE STORE

def _bootstrap_records(checkpoint_path: str, fallback_path: str) -> dict[int, dict]:
    """Reuse extractions from this run's checkpoint, or from v5's store."""
    for path in (checkpoint_path, fallback_path):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f).get("raw_records", [])
        except (json.JSONDecodeError, OSError):
            print(f"  ! {path} unreadable; skipping")
            continue
        done = {r["index"]: r for r in records
                if isinstance(r.get("index"), int) and r.get("stats")}
        if done:
            print(f"Reusing {len(done)} extractions from {path} "
                  f"(the schema is unchanged since v5).")
            return done
    return {}


def build_store(document_text: str, narrative_tree_path: str = "tree.json",
                max_workers: int = 6, checkpoint_path: str = "store_v6.json",
                fallback_store: str = "store_v5.json") -> dict:
    with timed("split document into entries"):
        entries = find_park_entries(document_text)
    print(f"Split document into {len(entries)} park entries.")
    if not entries:
        raise RuntimeError(
            "No 'Park in numbers' boxes found. This pipeline is anchored on "
            "that phrase -- use pipeline_v2.py for a document without it."
        )
    entry_by_index = {e["index"]: e["entry"] for e in entries}

    done = _bootstrap_records(checkpoint_path, fallback_store)
    todo = [e for e in entries if e["index"] not in done]
    if todo:
        with timed(f"extraction ({len(todo)} LLM calls, {max_workers} at a time)"):
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(extract_park, e): e["index"] for e in todo}
                for future in as_completed(futures):
                    record = future.result()
                    done[record["index"]] = record
                    print(f"[extract] {len(done)}/{len(entries)} "
                          f"({record.get('name') or 'unnamed'})")
    else:
        STAGE_SECONDS["extraction (reused, 0 LLM calls)"] = 0.0

    records = [done[i] for i in sorted(done)]
    parks = [normalize_park(r) for r in records]

    # --- figure validation (unchanged from v5: range check + bijection check)
    suspect = [
        (r, p) for r, p in zip(records, parks)
        if implausible(p) or stats_disagree_with_box(
            r.get("stats", []), r.get("box_numbers", []))
    ]
    if suspect:
        with timed(f"validation ({len(suspect)} re-reads)"):
            print(f"\n[validate] {len(suspect)} parks failed a check; re-reading.")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for record in pool.map(lambda s: recheck_park(*s), suspect):
                    done[record["index"]] = record
            records = [done[i] for i in sorted(done)]
            parks = [normalize_park(r) for r in records]

    for record, park in zip(records, parks):
        problems = implausible(park) + stats_disagree_with_box(
            record.get("stats", []), record.get("box_numbers", []))
        if problems:
            park["figures_unreliable"] = problems

    # --- fold the deterministic indexes onto each row
    with timed("deterministic indexes (no LLM)"):
        by_index = {r.get("index"): r for r in records}
        for park in parks:
            record = by_index.get(park.get("index"), {})
            entry_text = entry_by_index.get(park.get("index"), "")

            park.update(resolve_country(record.get("country"), entry_text))
            park["country_evidence"] = record.get("country_evidence")
            park["designations"] = sorted({
                canonical_designation(d)
                for d in (record.get("designations") or [])
            })
            park["cross_border"] = record.get("cross_border") or {}
            park["named_features"] = (record.get("named_features") or [])[:10]
            park["notable_facts"] = record.get("notable_facts") or []
            park["superlative_claims"] = merge_claims(
                record.get("superlative_claims") or [],
                scan_entry_superlatives(entry_text, park.get("name")),
            )
            # the engine reads claims off the record, so keep both in step
            record["superlative_claims"] = park["superlative_claims"]
            record["designations"] = park["designations"]

        # --- everything countable, counted in Python
        computed = compute_aggregates(parks)
        computed.update(extend_aggregates(parks, records))
        computed.pop("named_features_index", None)

        coverage = build_coverage_index(document_text)
        events = build_dated_events(parks)

        computed["contradictions"] = find_contradictions(parks, records, computed)
        computed["superlative_claims_index"] = [
            {"park": p.get("name"), **c}
            for p in parks for c in p.get("superlative_claims", [])
        ]
        computed["dated_events"] = events
        computed["firsts_and_earliest"] = build_firsts_index(events)
        computed["theme_index"] = build_theme_index(parks, coverage, entry_by_index)
        computed["countries_uncertain"] = [
            {"park": p.get("name"), "country": p.get("country"),
             "mentions": p.get("country_mentions")}
            for p in parks if p.get("country_uncertain")
        ]

    narratives = load_narratives(narrative_tree_path)
    # v2's topic_index is ~1,400 vague strings a 3B model produced while
    # summarizing chunks. The coverage index replaces it with exact counts over
    # the same document, and keeping both crowded the real evidence out of the
    # context window.
    narratives.pop("topic_index", None)

    store = {
        "parks": parks,
        "computed": computed,
        "coverage_index": coverage,
        "narratives": narratives,
        "raw_records": records,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    return store


# --------------------------------------------------------------- ANSWER

def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


def build_answer_context(store: dict, budget: int = ANSWER_TOKEN_BUDGET) -> dict:
    """
    Trim to fit the context window, by SIZE ONLY and in a fixed order.

    Nothing here consults the question. Every question receives byte-identical
    data, so no question ever gets a narrower view of the document than
    another -- the trimming is a context-window constraint, not retrieval.

    The coverage index is trimmed LAST and only as a true last resort: it is
    the sole evidence for absence questions, and a truncated one does not merely
    weaken an answer, it makes a discussed subject look absent.
    """
    def fits(context: dict) -> bool:
        return _estimate_tokens(_dumps(context)) <= budget

    def map_parks(context: dict, transform) -> dict:
        return {**context, "parks": [transform(dict(p)) for p in context["parks"]]}

    def cap_lists(context: dict) -> dict:
        def cap(park: dict) -> dict:
            park["notable_facts"] = (park.get("notable_facts") or [])[:8]
            park["named_features"] = (park.get("named_features") or [])[:5]
            park["superlative_claims"] = (park.get("superlative_claims") or [])[:4]
            return park
        return map_parks(context, cap)

    def strip_park_extras(context: dict) -> dict:
        def strip(park: dict) -> dict:
            park.pop("named_features", None)
            park.pop("stats", None)
            park.pop("country_mentions", None)
            return park
        return map_parks(context, strip)

    def drop_claims_index(context: dict) -> dict:
        # The verified findings in computed.contradictions stay; only the raw
        # 200+ claim dump goes.
        computed = dict(context.get("computed") or {})
        computed.pop("superlative_claims_index", None)
        return {**context, "computed": computed}

    def drop_dated_events(context: dict) -> dict:
        # firsts_and_earliest is a subset of dated_events, so the full list is
        # the redundant half -- and the facts also live on each park row.
        computed = dict(context.get("computed") or {})
        computed.pop("dated_events", None)
        return {**context, "computed": computed}

    def drop_sections(context: dict) -> dict:
        narratives = {k: v for k, v in (context.get("narratives") or {}).items()
                      if k != "sections"}
        return {**context, "narratives": narratives}

    def cap_firsts(context: dict) -> dict:
        computed = dict(context.get("computed") or {})
        computed["firsts_and_earliest"] = (computed.get("firsts_and_earliest") or [])[:150]
        return {**context, "computed": computed}

    def cap_facts(limit: int):
        def step(context: dict) -> dict:
            def trim(park: dict) -> dict:
                park["notable_facts"] = (park.get("notable_facts") or [])[:limit]
                return park
            return map_parks(context, trim)
        return step

    context = {k: v for k, v in store.items() if k != "raw_records"}
    if fits(context):
        return context

    for label, step in (
        ("capped per-park lists", cap_lists),
        ("dropped per-park features and raw stats", strip_park_extras),
        ("dropped the raw claims index (verified findings kept)", drop_claims_index),
        ("dropped the full dated-events list (firsts kept)", drop_dated_events),
        ("dropped section narratives", drop_sections),
        ("capped the firsts index to 150 entries", cap_firsts),
        ("capped per-park facts to 5", cap_facts(5)),
        ("capped per-park facts to 3", cap_facts(3)),
    ):
        context = step(context)
        if fits(context):
            print(f"  (answer context: {label})")
            return context

    # Only now touch the absence evidence, and shrink it as little as possible.
    for keep in (3500, 2500, 1500, 900, 500):
        reduced = dict(context.get("coverage_index") or {})
        terms = reduced.get("terms") or {}
        reduced["terms"] = dict(list(terms.items())[:keep])
        reduced["note"] = (reduced.get("note", "") +
                           f" TRUNCATED to the {keep} most frequent terms to fit "
                           f"the context window: a subject missing here may still "
                           f"appear in the document, so do not treat omission as "
                           f"proof of absence.")
        candidate = {**context, "coverage_index": reduced}
        if fits(candidate):
            print(f"  (answer context: coverage index reduced to {keep} terms)")
            return candidate
    return candidate


ANSWER_PROMPT = """Answer the question using ONLY the data below, compiled from a book of
European national parks. Do not use outside knowledge.

Everything here was built from the whole document during ingest. Every
question receives exactly this same data.

How to use each part:
- "computed" was calculated in code. For any counting, ranking, "how many" or
  "which is the largest/highest/most" question, TRUST COMPUTED OVER YOUR OWN
  COUNTING. Do not recount by hand.
- "computed.contradictions" lists conflicts ALREADY VERIFIED IN CODE between
  what the book claims in prose and what its own figures say. For any question
  about the book contradicting itself or being inconsistent, ANSWER FROM THIS
  LIST FIRST, naming both sides and their numbers.
- "coverage_index" holds exact word and phrase counts over the entire
  document, with anything appearing fewer than min_count times omitted. THIS
  IS THE EVIDENCE FOR ABSENCE QUESTIONS. A subject missing from this index, or
  present only at a very low count, is not substantively discussed -- say so
  explicitly and confidently. A subject with a high count is discussed. Check
  the index rather than reasoning from what you happen to remember reading.
- "computed.theme_index" maps recurring themes to the parks that discuss them,
  with counts. Use it for questions about what recurs across the book.
- "computed.dated_events" and "computed.firsts_and_earliest" list every fact
  carrying a year, sorted, with its park. Use these for questions about dates,
  firsts, and events recorded in entries far apart in the book.
- "computed.parks_by_designation" uses canonical designation names, so all
  spellings of a designation are already grouped.
- "parks" is the complete table, one row per park. A row carrying
  "figures_unreliable" had a figure that could not be read from the scrambled
  source box -- do not rank or count on that row's numbers. A row carrying
  "country_uncertain" may be attributed to the wrong country.
- "narratives" holds the ordered section summaries and the document's arc,
  for themes and development across the book.

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


def answer_question(store: dict, question: str) -> str:
    """
    Answer one question against the store.

    temperature=0 is not a tuning choice, it is what makes a score readable.
    Two runs of identical code over identical data scored 67.0% and 60.4%:
    every question backed by a computed value was bit-identical, and every
    free-prose question moved. Sampling noise that large drowns out most real
    improvements, so a change cannot be evaluated until it is removed.
    """
    prompt = ANSWER_PROMPT.format(
        store=_dumps(build_answer_context(store)), question=question
    )
    return call_llm(prompt, model=MODEL_ANSWER, temperature=ANSWER_TEMPERATURE).strip()


def run_submission(store: dict, questions: list[dict], output_path: str,
                   team: str = "group_c", notes: str = "",
                   ingest_seconds: float = 0.0) -> dict:
    """
    Answer every question, writing the submission after each one.

    Timing is recorded per question and written into the file alongside the
    answers. The grader ignores unknown fields, so this costs nothing on the
    upload and gives a record of where the wall clock went.
    """
    started = time.perf_counter()
    submission = {
        "team": team,
        "notes": notes,
        "answers": [],
        "timing": {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ingest_seconds": round(ingest_seconds, 2),
            "stages": {k: round(v, 2) for k, v in STAGE_SECONDS.items()},
        },
    }
    per_question: dict[str, float] = {}

    for q in questions:
        qid = q["id"]
        print(f"Answering {qid}...")
        entry = {"id": qid}
        question_start = time.perf_counter()
        try:
            entry["answer"] = answer_question(store, q["question"])
        except Exception as e:
            entry["answer"] = ""
            entry["error"] = str(e)
            print(f"  -> failed: {e}")

        elapsed = time.perf_counter() - question_start
        per_question[qid] = round(elapsed, 2)
        entry["seconds"] = round(elapsed, 2)
        print(f"  [{format_duration(elapsed):>9}] {qid}")

        submission["answers"].append(entry)
        answering = time.perf_counter() - started
        submission["timing"].update({
            "answering_seconds": round(answering, 2),
            "total_seconds": round(ingest_seconds + answering, 2),
            "per_question_seconds": dict(per_question),
        })
        # written every iteration, timings included: a partial file still beats
        # no file, and it still says how long it took to get that far
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

    answering = time.perf_counter() - started
    print(f"\nSubmission written to {output_path} "
          f"({len(submission['answers'])} answers)")
    if per_question:
        slowest = max(per_question.items(), key=lambda kv: kv[1])
        print(f"  answering:  {format_duration(answering)} total, "
              f"{format_duration(answering / len(per_question))} average, "
              f"slowest {slowest[0]} at {format_duration(slowest[1])}")
    return submission


if __name__ == "__main__":
    run_started = time.perf_counter()

    with timed("read and preprocess document"):
        with open("nationalparks_europe.txt", "r", encoding="utf-8") as f:
            document_text = preprocess_document(f.read())

    store = build_store(document_text, narrative_tree_path="tree.json")
    ingest_seconds = time.perf_counter() - run_started
    computed, coverage = store["computed"], store["coverage_index"]

    print(f"\n{'=' * 62}")
    print(f"context      ~= {_estimate_tokens(_dumps(build_answer_context(store))):,} tokens"
          f"  (budget {ANSWER_TOKEN_BUDGET:,})")
    print(f"parks           {computed['total_parks_profiled']} in "
          f"{len(computed['parks_by_country'])} countries")
    print(f"coverage index  {len(coverage['terms']):,} terms over "
          f"{coverage['total_words']:,} words")
    print(f"dated events    {len(computed['dated_events'])} "
          f"({len(computed['firsts_and_earliest'])} firsts)")
    print(f"themes          {len(computed['theme_index'])}")
    print(f"claims scanned  {len(computed['superlative_claims_index'])}")
    print(f"designations    {list(computed['parks_by_designation'])}")
    print(f"uncertain country: {[c['park'] for c in computed['countries_uncertain']]}")
    print(f"\nCONTRADICTIONS ({len(computed['contradictions'])}):")
    for finding in computed["contradictions"]:
        print(f"  [{finding['type']}] {finding['finding']}")

    # Sanity-check the absence machinery against the subjects d07-d09 name.
    print("\nCoverage spot-check:")
    for probe in ("poaching", "world heritage", "glacier", "glaciation",
                  "wartime", "military", "reindeer herding", "natura"):
        result = term_presence(coverage, probe)
        print(f"  {probe:<18} count={result['count']:<5} "
              f"substantive={result['substantive']}")
    print("=" * 62)

    print(f"\nIngest took {format_duration(ingest_seconds)}:")
    for stage, seconds in STAGE_SECONDS.items():
        share = 100 * seconds / ingest_seconds if ingest_seconds else 0
        print(f"  {format_duration(seconds):>9}  {share:4.1f}%  {stage}")
    print("=" * 62)

    questions = load_questions("dev_questions.yaml")
    submission = run_submission(
        store, questions, output_path="submission_v6.json", team="group_c",
        ingest_seconds=ingest_seconds,
        notes="Park table with validated figures, plus deterministic indexes "
              "built over the whole document at ingest: exact-count coverage "
              "index for absence, regex superlative scan cross-checked against "
              "the figures for contradictions, sorted dated-event index, "
              "canonical designations, and text-reconciled country attribution. "
              "All counting computed in Python; no per-question retrieval.",
    )

    total = time.perf_counter() - run_started
    timing = submission["timing"]
    print("=" * 62)
    print(f"TOTAL {format_duration(total)}"
          f"   ingest {format_duration(timing['ingest_seconds'])}"
          f" + answering {format_duration(timing['answering_seconds'])}")
    print(f"Timings are also recorded in submission_v6.json under \"timing\" "
          f"(ignored by the grader).")
    print("=" * 62)
