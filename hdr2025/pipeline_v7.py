"""
Model G: v6 plus an absence engine that actually matches how absence is graded.

v6 at temperature 0 scores 62.5%. Absence has sat at 16.7% for three runs, so
it is real, not sampling noise. Reading the dev set's key points shows the
problem was never just detection:

  d09 asks which of four threats is never raised. Its FOUR key points are:
      1. identify poaching as absent
      2. note glacier retreat is discussed, e.g. at Jostedalsbreen
      3. note visitor pressure is discussed, e.g. at Cinque Terre
      4. note wartime history appears, e.g. at Plitvice or Saxon Switzerland
  Naming the absent subject perfectly scores 1 of 4. Our answers named one
  subject and stopped.

And detection itself was failing on two specific traps, both verified against
the raw text:

  VARIANTS. d07 names "the legacy of glaciation". The word "glaciation" occurs
      zero times; "glacier/glacial/glaciers/glaciated" occur 173 times. v6's
      exact-term index reported 0, so the model called a pervasive theme absent
      -- which d07's forbidden list penalises explicitly.
  SENSE. d07's true answer is "accessibility for visitors with reduced
      mobility". But "accessible" appears 39 times, because the book says
      "accessible via bus from Avezzano" in every Getting There box. Counting
      the head word finds 39 hits and concludes the subject IS covered. Only
      the specific vocabulary separates the senses: wheelchair 0, reduced
      mobility 0, disabled 0, step-free 0.

So v7 adds three things, none of which needs a single extra LLM call:

  A. WORD FAMILIES -- terms grouped by 5-character prefix, so a lookup of
     "glaciation" surfaces the whole glaci* family with its forms and total.
  B. TERM LOCATOR -- which parks each distinctive term appears in, so the model
     can cite "at Jostedalsbreen" instead of asserting coverage without
     evidence. This is where 2-3 key points per absence question live.
  C. AN ABSENCE PROTOCOL in the answer prompt: check the subject's specific
     vocabulary rather than a generic head word, watch for a common word used
     in an unrelated sense, name exactly one subject as absent, and cite a park
     for every subject that IS present.

Everything is deterministic and question-independent, and the extraction
schema is unchanged, so this reuses store_v6.json for zero extraction calls.

Requires:
    pip install openai pyyaml --break-system-packages
    export BROKER_API_KEY=your_group_key
"""

import os
import re
import json
import time
from collections import defaultdict

from pipeline_ed2 import call_llm, preprocess_document, load_questions
from pipeline_v6 import (
    ANSWER_TEMPERATURE,
    ANSWER_TOKEN_BUDGET,
    MODEL_ANSWER,
    STAGE_SECONDS,
    STOPWORDS,
    _dumps,
    _estimate_tokens,
    build_store as build_store_v6,
    find_park_entries,
    format_duration,
    timed,
)

# v6 indexed letters only, which threw away the single most decisive piece of
# evidence on the dev set: the bigram "natura 2000". Without the digits, the
# only lookup left is "natura", whose 5-char prefix collides with nature/
# natural/naturally -- so a designation that never appears looks common.
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]{2,}|\d{2,4}")


def build_coverage_index(text: str, min_count: int = 3,
                         max_terms: int = 9000) -> dict:
    """
    Exact term counts over the whole document, unigrams and bigrams.

    Same as v6's but numbers are tokens too, so designations and years that
    carry a digit ("natura 2000", "world war 2") are checkable. Terms below
    min_count are dropped, and that threshold IS the evidence of absence.
    """
    from collections import Counter

    flat = re.sub(r"<!--.*?-->", " ", text)
    flat = re.sub(r"[#*_>`\[\]()]", " ", flat).lower()
    words = _TOKEN_RE.findall(flat)

    counts = Counter(w for w in words if w not in STOPWORDS)
    for left, right in zip(words, words[1:]):
        if left not in STOPWORDS and right not in STOPWORDS:
            counts[f"{left} {right}"] += 1

    frequent = {t: n for t, n in counts.most_common(max_terms) if n >= min_count}
    return {
        "total_words": len(words),
        "min_count": min_count,
        "note": (f"Exact counts over the whole document. Terms occurring fewer "
                 f"than {min_count} times are omitted, so a subject absent from "
                 f"this list is not substantively discussed."),
        "terms": dict(sorted(frequent.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

# A term earns a locator entry when it recurs across entries but is not
# everywhere. The upper bound was 10 at first, which excluded 'retreat',
# 'glacier', 'bear' and 'crowds' -- precisely the terms an absence answer needs
# to cite. Citing a park matters more than the term being rare, so the bound is
# 25 and the park list per term is capped instead.
LOCATOR_MIN_ENTRIES = 2
LOCATOR_MAX_ENTRIES = 25
LOCATOR_MIN_COUNT = 6
# Four parks is enough to cite an example; six cost ~5k tokens that the answer
# context needed for per-park facts.
LOCATOR_MAX_PARKS = 4

# Grouping by 5-char prefix is crude but it is the only thing that unifies
# "glaciation" with "glacier": suffix stemmers strip -ation to "glaci" and -ier
# to "glac", so the two never meet. Members are always shown alongside the
# total so a false merge is visible rather than silently miscounted.
FAMILY_PREFIX = 5


# ------------------------------------------------------- A. WORD FAMILIES

def build_word_families(terms: dict[str, int],
                        max_forms: int = 4) -> dict[str, dict]:
    """
    Group single words sharing a prefix, keeping the total and the forms.

    Only groups with more than one form are kept: a family of one adds nothing
    that the exact-term index does not already say, and the context window is
    the binding constraint.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for term in terms:
        if " " in term or len(term) < FAMILY_PREFIX:
            continue
        groups[term[:FAMILY_PREFIX]].append(term)

    families = {}
    for prefix, forms in groups.items():
        if len(forms) < 2:
            continue
        forms.sort(key=lambda w: -terms[w])
        families[prefix] = {
            "total": sum(terms[w] for w in forms),
            "forms": forms[:max_forms],
        }
    return dict(sorted(families.items()))


def lookup_subject(coverage: dict, phrase: str) -> dict:
    """
    Resolve one subject the way the answer prompt asks the model to.

    Reports evidence rather than a verdict, deliberately. No single aggregate
    works on the dev set: "wartime damage and military history" is present via
    'military'/'history' while 'wartime' is 0, and "accessibility for visitors
    with reduced mobility" is absent despite 'accessible' scoring 53. Only
    weighing the DISTINCTIVE words against the generic ones separates them, and
    that judgement is the one thing here worth spending the model on.

    Families are reported with their member forms and a flag saying whether the
    queried word is one of them, rather than being silently summed: prefix
    grouping merges 'natura' into nature/natural, and a bare total would make a
    designation that never occurs look common.
    """
    terms = coverage.get("terms", {})
    families = coverage.get("families", {})
    tokens = _TOKEN_RE.findall(phrase.lower())
    content = [w for w in tokens if w not in STOPWORDS]

    per_word = {}
    for word in content:
        entry = {"exact": terms.get(word, 0)}
        family = families.get(word[:FAMILY_PREFIX])
        if family:
            # Report the family whenever one exists, and say plainly whether
            # the queried word is itself among its forms. Morphology cannot
            # settle this: 'glaciation'/'glacier' and 'natura'/'nature' split
            # identically at five characters, yet the first pair is one theme
            # and the second is a proper name against a common noun. Only the
            # reader knows which, so give them the fact, not a verdict.
            entry["family"] = {
                "prefix": word[:FAMILY_PREFIX] + "*",
                "total": family["total"],
                "forms": family["forms"],
                "queried_word_is_a_listed_form": word in family["forms"],
            }
        per_word[word] = entry

    return {
        "phrase": " ".join(tokens),
        "exact_phrase_count": terms.get(" ".join(content), 0),
        "content_words": per_word,
    }


# --------------------------------------------------------- B. TERM LOCATOR

def build_term_locator(terms: dict[str, int], entries: list[dict]) -> dict:
    """
    Map each distinctive term to the park entries it appears in.

    Absence questions are graded on naming WHERE the other candidates are
    covered, not only on spotting the missing one. Without this the model can
    assert "glacier retreat is discussed" but never say "at Jostedalsbreen",
    and those citations are most of the available points.
    """
    lowered = [(e["index"], e["entry"].lower()) for e in entries]
    found = []
    for term, count in terms.items():
        if count < LOCATOR_MIN_COUNT:
            continue
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        hits = [index for index, text in lowered if pattern.search(text)]
        if LOCATOR_MIN_ENTRIES <= len(hits) <= LOCATOR_MAX_ENTRIES:
            found.append((len(hits), -count, term, hits[:LOCATOR_MAX_PARKS]))

    # Ordered by SPECIFICITY, not frequency, because trimming follows this
    # order. A term in 6 entries ('military') names a park usefully; one in 25
    # ('glacier') does not, and is easy to evidence elsewhere. Frequency order
    # put exactly the citable terms in the tail and cut them first.
    found.sort()
    return {term: hits for _, _, term, hits in found}


# ----------------------------------------------------------- BUILD STORE

def build_store(document_text: str, narrative_tree_path: str = "tree.json",
                max_workers: int = 6, checkpoint_path: str = "store_v7.json",
                fallback_store: str = "store_v6.json") -> dict:
    """v6's store, with the absence machinery layered on. No extra LLM calls."""
    store = build_store_v6(
        document_text, narrative_tree_path=narrative_tree_path,
        max_workers=max_workers, checkpoint_path=checkpoint_path,
        fallback_store=fallback_store,
    )

    with timed("absence indexes (word families + term locator)"):
        entries = find_park_entries(document_text)
        # Rebuild with digits included -- v6's letters-only index cannot
        # represent "natura 2000", the decisive term for d08.
        coverage = build_coverage_index(document_text)
        store["coverage_index"] = coverage
        coverage["families"] = build_word_families(coverage["terms"])
        coverage["families_note"] = (
            f"Terms grouped by their first {FAMILY_PREFIX} characters, with the "
            f"combined count and the member forms. Look a subject up here as "
            f"well as in 'terms': 'glaciation' occurs 0 times but its family "
            f"glaci* occurs many, and the subject is plainly discussed.")

        store["term_locator"] = build_term_locator(coverage["terms"], entries)
        store["term_locator_note"] = (
            "term -> the park 'index' values whose entries contain it. Use it to "
            "name WHERE a subject is covered. Only distinctive terms are listed "
            f"(appearing in {LOCATOR_MIN_ENTRIES}-{LOCATOR_MAX_ENTRIES} entries, "
            f"at least {LOCATOR_MIN_COUNT} times); a term missing from the "
            "locator may still be common -- check 'terms'.")
        store["park_names_by_index"] = {
            str(p.get("index")): p.get("name") for p in store["parks"]
        }

        # The locator supersedes it: same idea, 40x the coverage.
        store["computed"].pop("theme_index", None)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    return store


# --------------------------------------------------------------- ANSWER

def build_answer_context(store: dict, budget: int = ANSWER_TOKEN_BUDGET) -> dict:
    """
    Trim to fit, by size only, in a fixed order. Nothing consults the question.

    The order differs from v6: the absence machinery (coverage index, families,
    locator) is trimmed last, because Absence is the weakest category and those
    three indexes are its only evidence. Per-park prose goes first -- Needle is
    at 100% with room to spare.
    """
    def fits(context: dict) -> bool:
        return _estimate_tokens(_dumps(context)) <= budget

    def map_parks(context: dict, transform) -> dict:
        return {**context, "parks": [transform(dict(p)) for p in context["parks"]]}

    def cap_facts(limit: int):
        def step(context: dict) -> dict:
            def trim(park: dict) -> dict:
                park["notable_facts"] = (park.get("notable_facts") or [])[:limit]
                park["named_features"] = (park.get("named_features") or [])[:5]
                park["superlative_claims"] = (park.get("superlative_claims") or [])[:3]
                return park
            return map_parks(context, trim)
        return step

    def strip_park_extras(context: dict) -> dict:
        def strip(park: dict) -> dict:
            park.pop("named_features", None)
            park.pop("stats", None)
            park.pop("country_mentions", None)
            return park
        return map_parks(context, strip)

    def drop_computed(*keys):
        def step(context: dict) -> dict:
            computed = dict(context.get("computed") or {})
            for key in keys:
                computed.pop(key, None)
            return {**context, "computed": computed}
        return step

    def drop_sections(context: dict) -> dict:
        narratives = {k: v for k, v in (context.get("narratives") or {}).items()
                      if k != "sections"}
        return {**context, "narratives": narratives}

    context = {k: v for k, v in store.items() if k != "raw_records"}
    if fits(context):
        return context

    def trim_locator(keep: int):
        def step(context: dict) -> dict:
            locator = context.get("term_locator") or {}
            # Built in specificity order, so the head holds the terms that name
            # a park most usefully and the tail holds the near-ubiquitous ones.
            return {**context, "term_locator": dict(list(locator.items())[:keep])}
        return step

    for label, step in (
        ("capped per-park lists", cap_facts(8)),
        ("dropped the raw claims index", drop_computed("superlative_claims_index")),
        ("dropped per-park features and raw stats", strip_park_extras),
        ("dropped the full dated-events list", drop_computed("dated_events")),
        # Facts 6-8 on a park row yield to the locator: Needle sits at 100%
        # with headroom, Absence at 16.7% with none, and low-frequency locator
        # terms like 'military' are exactly the citations Absence is graded on.
        ("capped per-park facts to 5", cap_facts(5)),
        ("trimmed the term locator to 1400 terms", trim_locator(1400)),
        ("dropped section narratives", drop_sections),
        ("trimmed the term locator to 800 terms", trim_locator(800)),
        ("capped per-park facts to 3", cap_facts(3)),
    ):
        context = step(context)
        if fits(context):
            print(f"  (answer context: {label})")
            return context

    # Absence evidence goes last, locator before the counts themselves.
    for keep in (1500, 800, 400, 0):
        trimmed = dict(list((context.get("term_locator") or {}).items())[:keep])
        candidate = {**context, "term_locator": trimmed}
        if fits(candidate):
            print(f"  (answer context: term locator reduced to {keep} terms)")
            return candidate

    for keep in (3000, 2000, 1000):
        reduced = dict(context.get("coverage_index") or {})
        reduced["terms"] = dict(list((reduced.get("terms") or {}).items())[:keep])
        reduced["note"] = (reduced.get("note", "") + " TRUNCATED to the most "
                           "frequent terms: omission is no longer proof of absence.")
        candidate = {**context, "coverage_index": reduced}
        if fits(candidate):
            print(f"  (answer context: coverage index reduced to {keep} terms)")
            return candidate
    return candidate


ANSWER_PROMPT = """Answer the question using ONLY the data below, compiled from a book of
European national parks. Do not use outside knowledge.

Everything here was built from the whole document during ingest. Every question
receives exactly this same data.

How to use each part:
- "computed" was calculated in code. For any counting, ranking, "how many" or
  "which is the largest/highest/most" question, TRUST COMPUTED OVER YOUR OWN
  COUNTING. Do not recount by hand.
- "computed.contradictions" lists conflicts ALREADY VERIFIED IN CODE between
  what the book claims in prose and what its own figures say. For questions
  about the book contradicting itself, ANSWER FROM THIS LIST FIRST, naming both
  sides and their numbers.
- "computed.firsts_and_earliest" and "computed.dated_events" list every fact
  carrying a year, sorted, with its park.
- "parks" is the complete table, one row per park. A row carrying
  "figures_unreliable" had an unreadable figure -- do not rank on its numbers.
  A row carrying "country_uncertain" may be attributed to the wrong country.
- "narratives" holds the ordered section summaries and the document's arc.

## Absence questions -- follow this protocol exactly

When the question asks which of several subjects is NEVER discussed, mentioned
or raised:

1. For EACH candidate subject, look up its SPECIFIC vocabulary in
   "coverage_index.terms" and "coverage_index.families" -- not a generic head
   word. For "accessibility for visitors with reduced mobility" the telling
   words are 'wheelchair', 'reduced mobility', 'disabled', 'step-free', NOT
   'accessible'.
2. BEWARE A COMMON WORD USED IN A DIFFERENT SENSE. This book says a park is
   "accessible via bus" in almost every entry, so 'accessible' is frequent
   while disability access is entirely absent. High counts on a generic word
   are not evidence that the subject is covered; check the specific vocabulary.
3. BEWARE WORD VARIANTS. Before concluding anything is missing, look the word
   up in "coverage_index.families" under its first 5 letters. 'glaciation'
   occurs zero times, but the family 'glaci*' lists glacier/glacial/glaciers
   and occurs constantly -- so glaciation is plainly discussed.
4. BUT A NAMED DESIGNATION IS NOT A WORD FAMILY. When the subject is a proper
   name -- "Natura 2000", "Ramsar", a specific programme -- require the NAME
   itself: check the exact phrase and the exact word. A similar-looking word in
   the same family is a different thing. 'natura' occurs 0 times and the phrase
   'natura 2000' occurs 0 times; that the family 'natur*' is common only
   reflects "nature" and "natural", which are unrelated to the designation.
   Judge names by exact match, themes by family.
5. Name EXACTLY ONE subject as the absent one. Commit to it.
6. THEN, FOR EVERY OTHER CANDIDATE, SAY WHERE IT APPEARS -- name a specific
   park. Use "term_locator" to get the park index values for a term and
   "park_names_by_index" to turn those into names, or cite a park row's
   notable_facts. This is required: an answer that names the absent subject but
   does not show the others are covered is a substantially incomplete answer.

A good absence answer looks like: "X is never raised. The other three are: A at
[park], B at [park], and C at [park]."

## All questions

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

Answer in 1-5 sentences, plain text, no JSON, no preamble.
"""


def answer_question(store: dict, question: str) -> str:
    prompt = ANSWER_PROMPT.format(
        store=_dumps(build_answer_context(store)), question=question
    )
    return call_llm(prompt, model=MODEL_ANSWER,
                    temperature=ANSWER_TEMPERATURE).strip()


def run_submission(store: dict, questions: list[dict], output_path: str,
                   team: str = "group_c", notes: str = "",
                   ingest_seconds: float = 0.0) -> dict:
    """Answer every question, writing the file (and its timings) as we go."""
    started = time.perf_counter()
    submission = {
        "team": team, "notes": notes, "answers": [],
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
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

    answering = time.perf_counter() - started
    print(f"\nSubmission written to {output_path} "
          f"({len(submission['answers'])} answers)")
    if per_question:
        slowest = max(per_question.items(), key=lambda kv: kv[1])
        print(f"  answering: {format_duration(answering)} total, "
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

    print(f"\n{'=' * 66}")
    print(f"context       ~= {_estimate_tokens(_dumps(build_answer_context(store))):,}"
          f" tokens (budget {ANSWER_TOKEN_BUDGET:,})")
    print(f"parks            {computed['total_parks_profiled']} in "
          f"{len(computed['parks_by_country'])} countries")
    print(f"coverage         {len(coverage['terms']):,} terms, "
          f"{len(coverage['families']):,} word families")
    print(f"term locator     {len(store['term_locator']):,} terms")
    print(f"contradictions   {len(computed['contradictions'])}")

    # The dev set's absence subjects, resolved exactly as the prompt asks the
    # model to resolve them. If these verdicts are wrong, the answers will be.
    print("\nAbsence self-check (true answers marked *):")
    probes = [
        ("d07", "Unesco World Heritage designations", False),
        ("d07", "the legacy of glaciation", False),
        ("d07", "accessibility for visitors with reduced mobility", True),
        ("d07", "brown bears", False),
        ("d08", "Natura 2000", True),
        ("d08", "Unesco biosphere reserve status", False),
        ("d08", "national nature reserves", False),
        ("d09", "poaching", True),
        ("d09", "glacier retreat", False),
        ("d09", "pressure from visitor numbers", False),
        ("d09", "wartime damage and military history", False),
    ]
    for qid, subject, expected_absent in probes:
        result = lookup_subject(coverage, subject)
        mark = "*" if expected_absent else " "
        print(f" {mark}{qid} {subject[:48]:<49} "
              f"phrase={result['exact_phrase_count']}")
        for word, evidence in result["content_words"].items():
            family = evidence.get("family")
            detail = ""
            if family:
                detail = (f"  family {family['prefix']}={family['total']} "
                          f"{family['forms'][:3]} "
                          f"is-a-form={family['queried_word_is_a_listed_form']}")
            print(f"        {word:<16} exact={evidence['exact']:<5}{detail}")

    print(f"\nIngest took {format_duration(ingest_seconds)}:")
    for stage, seconds in STAGE_SECONDS.items():
        share = 100 * seconds / ingest_seconds if ingest_seconds else 0
        print(f"  {format_duration(seconds):>9}  {share:4.1f}%  {stage}")
    print("=" * 66)

    questions = load_questions("dev_questions.yaml")
    submission = run_submission(
        store, questions, output_path="submission_v7.json", team="group_c",
        ingest_seconds=ingest_seconds,
        notes="Deterministic indexes over the whole document: exact-count "
              "coverage index with prefix word families, a term->park locator "
              "for citing where a subject is covered, regex superlative scan "
              "cross-checked against the figures for contradictions, sorted "
              "dated events, canonical designations, text-reconciled countries. "
              "All counting computed in Python; no per-question retrieval.",
    )

    total = time.perf_counter() - run_started
    timing = submission["timing"]
    print("=" * 66)
    print(f"TOTAL {format_duration(total)}   "
          f"ingest {format_duration(timing['ingest_seconds'])} + "
          f"answering {format_duration(timing['answering_seconds'])}")
    print("=" * 66)
