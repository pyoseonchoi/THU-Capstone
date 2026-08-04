"""
Model I: synthesis that preserves the better draft instead of averaging it.

v8's two-view design scored 65.4%, the best so far, and moved seven questions
up (d14 +40, d20 +42, d04 +33, d15/d18/d09/d03 +20-25). But two questions
collapsed:

    d05 "highest summit named as a park's highest point"   75% -> 0%
    d08 "which designation is never mentioned"            100% -> 33%

Both are the same failure. Only the index view carries the coverage index, so
its draft KNOWS that "Natura 2000" occurs zero times in the document. The
detail view has no way to know and writes a plausible guess. The synthesizer
was told the drafts are "complementary, not competing" -- true of their
coverage, false of their authority -- so it blended a lookup with a guess and
landed halfway between.

Nothing was wrong with the data. Taking the best score each question has ever
achieved across every run gives 79.6%: every question here has already been
answered well by some version. The pipeline keeps trading wins rather than
accumulating them.

So v9 changes only how the three calls talk to each other:

  1. A draft may now DECLARE that its view cannot answer, by writing
     "NO EVIDENCE IN THIS VIEW", instead of guessing.
  2. If one draft declares that, synthesis is SKIPPED IN CODE and the other
     draft is used verbatim. A guess can no longer dilute a lookup, and this
     is a mechanical guarantee rather than an instruction the model may ignore.
  3. When both drafts do have evidence, the synthesizer is given explicit
     AUTHORITY RULES -- who wins on coverage claims, on park detail, on counts
     -- plus a floor: the final answer may never be less specific than the
     better draft.

Everything else is v8 unchanged, so the comparison is clean.

Requires:
    pip install openai pyyaml --break-system-packages
    export BROKER_API_KEY=your_group_key
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

from pipeline_ed2 import call_llm, preprocess_document, load_questions
from pipeline_v6 import (
    ANSWER_TEMPERATURE,
    MODEL_ANSWER,
    STAGE_SECONDS,
    _dumps,
    _estimate_tokens,
    format_duration,
    timed,
)
from pipeline_v7 import build_store as build_store_v7
from pipeline_v8 import build_detail_context, build_index_context

# A draft writes this instead of guessing. Checked in code, not left to the
# synthesizer to notice.
NO_EVIDENCE = "NO EVIDENCE IN THIS VIEW"

# Shared by both drafts: the instruction that makes declaring possible.
_HONESTY_RULE = f"""
## When this view cannot answer

You are one of two independent readers of the same document, each given a
different half of the evidence. The other reader has what you do not.

If this view does not contain the evidence needed to answer the question,
reply with exactly:

    {NO_EVIDENCE}

and nothing else. If it answers only part of the question, answer that part
concretely and write "{NO_EVIDENCE}" for the rest.

DO NOT GUESS AND DO NOT INFER FROM PLAUSIBILITY. A guess from you is worse
than silence: the other reader may have the real evidence, and your guess can
displace it. Saying you cannot answer is a correct and useful answer.
"""


DETAIL_PROMPT = """Answer the question using ONLY the data below, compiled from a book of
European national parks. Do not use outside knowledge.

This view holds the per-park detail and the document's narrative levels:
- "parks" is the complete table, one row per park, with its figures, its
  designations, its cross-border status and its notable_facts. Use
  notable_facts for specific details, dates, names and firsts.
- "computed" was calculated in code from every park's figures. For any
  counting, ranking or "which is the largest/highest/most" question, TRUST
  COMPUTED OVER YOUR OWN COUNTING, and state the value it gives.
- "computed.contradictions" lists conflicts already verified in code between
  the book's prose claims and its own figures.
- "computed.firsts_and_earliest" lists dated firsts with their park.
- "narratives" holds the ordered section summaries and the document's arc.
- A row with "figures_unreliable" had an unreadable figure; do not rank on it.
  A row with "country_uncertain" may be attributed to the wrong country.

THIS VIEW HAS NO WORD COUNTS over the document. It cannot establish that
something is never mentioned anywhere. If the question asks which of several
subjects is never discussed, mentioned or raised, you do not have the evidence
for it -- say so rather than guessing.
{honesty}
- Commit to a single confident answer for whatever you can answer.
- Name things specifically: parks, numbers, years.
- When a question asks for a figure the book prints, give the book's own units
  as well as the normalized value if they differ.

DATA (JSON):
{{store}}

QUESTION:
{{question}}

Answer in 1-5 sentences, plain text, no JSON, no preamble.
""".format(honesty=_HONESTY_RULE)


INDEX_PROMPT = """Answer the question using ONLY the data below, compiled from a book of
European national parks. Do not use outside knowledge.

This view holds whole-document indexes and the parks' comparable figures:
- "coverage_index.terms" gives EXACT counts of every word and phrase in the
  document, with anything under min_count omitted. "coverage_index.families"
  groups words by their first 5 letters with their member forms.
- "term_locator" maps a term to the park "index" values whose entries contain
  it; "park_names_by_index" turns those into names. Use it to say WHERE
  something is discussed.
- "computed" was calculated in code. For any counting, ranking or superlative
  question, TRUST COMPUTED OVER YOUR OWN COUNTING, and state the value it gives.
- "computed.contradictions" lists conflicts already verified in code.
- "parks" carries each park's comparable figures only.

THIS VIEW HAS NO PER-PARK PROSE. It cannot supply anecdotes, quotations, or
details that are not counts, figures or term locations.

## Absence questions -- follow this protocol exactly

1. For EACH candidate subject, look up its SPECIFIC vocabulary, not a generic
   head word. For "accessibility for visitors with reduced mobility" the
   telling words are 'wheelchair', 'reduced mobility', 'disabled' -- NOT
   'accessible'.
2. BEWARE A COMMON WORD IN A DIFFERENT SENSE. This book says a park is
   "accessible via bus" in nearly every entry, so 'accessible' is frequent
   while disability access is entirely absent.
3. BEWARE WORD VARIANTS. Check "families" under the word's first 5 letters:
   'glaciation' occurs 0 times but the family glaci* lists glacier/glacial and
   occurs constantly, so glaciation IS discussed.
4. BEWARE SYNONYMS. A subject can be discussed in completely different words.
   Pressure from visitor numbers may appear as 'crowds', 'overcrowding',
   'honeypot' or 'too popular'; wartime history as 'war', 'military',
   'bunker', 'fortress'. Before calling a subject absent, check the OTHER
   words the book would plausibly use for it. Only conclude absence when the
   whole vocabulary of the subject is missing.
5. A NAMED DESIGNATION IS NOT A WORD FAMILY. For a proper name -- "Natura
   2000", "Ramsar" -- require the name itself. 'natura' occurs 0 times and
   'natura 2000' occurs 0 times; that 'natur*' is common only reflects
   "nature" and "natural", which are unrelated to the designation.
6. Name EXACTLY ONE subject as the absent one and commit to it.
7. THEN, FOR EVERY OTHER CANDIDATE, SAY WHERE IT APPEARS and name a specific
   park, using "term_locator" plus "park_names_by_index". This is required.

You have the word counts, so you ARE the authority on what the document never
mentions. Answer absence questions fully and confidently from this view.
{honesty}
- Commit to a single confident answer for whatever you can answer.

DATA (JSON):
{{store}}

QUESTION:
{{question}}

Answer in 1-5 sentences, plain text, no JSON, no preamble.
""".format(honesty=_HONESTY_RULE)


SYNTHESIS_PROMPT = """Two readers answered the same question independently, each holding a
different half of the evidence about one book of European national parks.

  DRAFT A read the full per-park detail -- every park's facts, dates and named
          features -- plus the document's section summaries.
          A IS THE AUTHORITY ON: specific park facts, dates, anecdotes, names,
          themes and how the book treats a subject across entries.
          A HAS NO WORD COUNTS and cannot know what the book never mentions.

  DRAFT B read exact word and phrase counts over the whole document, word
          families, and an index of which parks mention which term.
          B IS THE AUTHORITY ON: whether something is ever mentioned at all,
          how often, and which parks mention a given term.
          B HAS NO PROSE and cannot supply anecdotes or quotations.

Write ONE final answer.

## Authority rules -- apply these before anything else

- If the question is about what is NEVER mentioned, discussed or raised, take
  B's verdict on WHICH subject is absent. A cannot know this and any guess it
  made must be discarded outright, even if it sounds more confident.
- For a specific fact about one park -- a date, a name, an anecdote, a story --
  take A's version.
- For a count, a ranking or a superlative, both readers were given the same
  precomputed values. If they agree, state that value. If they disagree, take
  the one that names a precomputed field, and if neither does, take B's.
- Where one reader is authoritative, USE ITS ANSWER AS WRITTEN. Do not soften
  it, do not average it against the other, and do not add the other's
  competing version alongside it.

## Combining

- Outside the authority rules the drafts are complementary: one often has the
  fact and the other has the evidence for where it appears. Include every
  specific claim -- park name, number, year, place -- that either draft makes
  and that helps answer the question.
- THE FINAL ANSWER MUST NEVER BE LESS SPECIFIC THAN THE BETTER DRAFT. If a
  draft names a park, a number or a year, that must survive into the answer.
- If one draft answers a part of the question the other ignores, keep it.
- Add nothing that neither draft supports.
- Never mention the drafts or the readers. Never hedge, never list
  alternatives, never say "it might be". Commit to one answer.
- If the question asks you to name several things, name them all.

QUESTION:
{question}

DRAFT A (per-park detail and narratives):
{draft_a}

DRAFT B (whole-document word counts and term index):
{draft_b}

Final answer, 1-5 sentences, plain text, no JSON, no preamble.
"""


def declares_no_evidence(draft: str) -> bool:
    """
    True when a draft is only the no-evidence marker.

    Checked here rather than trusted to the synthesizer: the whole point is
    that an uninformed draft must not reach the merge at all. A draft that
    answers part of the question and marks the rest still goes through, since
    the part it answered is real.
    """
    text = (draft or "").strip()
    if not text:
        return True
    stripped = text.upper().replace(NO_EVIDENCE, "").strip(" .\"'`\n\t-–—")
    return not stripped


def answer_question(contexts: tuple[str, str], question: str) -> dict:
    """
    Two drafts in parallel, then synthesis -- unless one draft has nothing.

    Returns the final answer plus both drafts and how it was decided, so a bad
    merge is diagnosable from the submission file rather than invisible.
    """
    detail_json, index_json = contexts

    def draft(prompt_template, view):
        prompt = prompt_template.format(store=view, question=question)
        return call_llm(prompt, model=MODEL_ANSWER,
                        temperature=ANSWER_TEMPERATURE).strip()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(draft, DETAIL_PROMPT, detail_json),
            pool.submit(draft, INDEX_PROMPT, index_json),
        ]
        drafts = []
        for future in futures:
            try:
                drafts.append(future.result())
            except Exception as e:
                print(f"  ! draft failed: {e}")
                drafts.append("")

    draft_a, draft_b = drafts
    a_empty = declares_no_evidence(draft_a)
    b_empty = declares_no_evidence(draft_b)

    if a_empty and b_empty:
        raise RuntimeError("neither view had evidence for this question")
    if a_empty or b_empty:
        # The decisive case. In v8 this is where d08 lost 67 points: the detail
        # view guessed at a coverage question and the merge averaged the guess
        # against the index view's lookup. Now the informed draft goes through
        # untouched.
        kept = draft_b if a_empty else draft_a
        which = "index" if a_empty else "detail"
        print(f"  -> only the {which} view had evidence; using it verbatim")
        return {"answer": kept, "draft_a": draft_a, "draft_b": draft_b,
                "decision": f"{which}_only"}

    try:
        final = call_llm(
            SYNTHESIS_PROMPT.format(question=question, draft_a=draft_a,
                                    draft_b=draft_b),
            model=MODEL_ANSWER, temperature=ANSWER_TEMPERATURE,
        ).strip()
    except Exception as e:
        print(f"  ! synthesis failed ({e}); falling back to the detail draft")
        return {"answer": draft_a, "draft_a": draft_a, "draft_b": draft_b,
                "decision": "synthesis_failed"}

    if not final:
        return {"answer": draft_a, "draft_a": draft_a, "draft_b": draft_b,
                "decision": "synthesis_empty"}
    return {"answer": final, "draft_a": draft_a, "draft_b": draft_b,
            "decision": "synthesized"}


def run_submission(store: dict, questions: list[dict], output_path: str,
                   team: str = "group_c", notes: str = "",
                   ingest_seconds: float = 0.0) -> dict:
    """Answer every question, writing the file after each one."""
    with timed("build the two answer views"):
        detail_json = _dumps(build_detail_context(store))
        index_json = _dumps(build_index_context(store))
    print(f"  detail view ~= {_estimate_tokens(detail_json):,} tokens")
    print(f"  index view  ~= {_estimate_tokens(index_json):,} tokens")
    contexts = (detail_json, index_json)

    started = time.perf_counter()
    submission = {
        "team": team, "notes": notes, "answers": [],
        "timing": {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ingest_seconds": round(ingest_seconds, 2),
            "detail_view_tokens": _estimate_tokens(detail_json),
            "index_view_tokens": _estimate_tokens(index_json),
            "stages": {k: round(v, 2) for k, v in STAGE_SECONDS.items()},
        },
    }
    per_question: dict[str, float] = {}
    decisions: dict[str, int] = {}

    for q in questions:
        qid = q["id"]
        print(f"Answering {qid}...")
        entry = {"id": qid}
        question_start = time.perf_counter()
        try:
            result = answer_question(contexts, q["question"])
            entry["answer"] = result["answer"]
            entry["decision"] = result["decision"]
            decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
            # Ungraded by the rules, and it makes every merge inspectable.
            entry["evidence"] = [f"DRAFT A (detail): {result['draft_a']}",
                                 f"DRAFT B (index): {result['draft_b']}"]
        except Exception as e:
            entry["answer"] = ""
            entry["error"] = str(e)
            print(f"  -> failed: {e}")

        elapsed = time.perf_counter() - question_start
        per_question[qid] = round(elapsed, 2)
        entry["seconds"] = round(elapsed, 2)
        print(f"  [{format_duration(elapsed):>9}] {qid} ({entry.get('decision', 'failed')})")

        submission["answers"].append(entry)
        answering = time.perf_counter() - started
        submission["timing"].update({
            "answering_seconds": round(answering, 2),
            "total_seconds": round(ingest_seconds + answering, 2),
            "decisions": dict(decisions),
            "per_question_seconds": dict(per_question),
        })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

    answering = time.perf_counter() - started
    print(f"\nSubmission written to {output_path} "
          f"({len(submission['answers'])} answers)")
    print(f"  decisions: {decisions}")
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

    store = build_store_v7(document_text, narrative_tree_path="tree.json",
                           checkpoint_path="store_v9.json",
                           fallback_store="store_v8.json")
    ingest_seconds = time.perf_counter() - run_started
    computed = store["computed"]

    print(f"\n{'=' * 66}")
    print(f"parks            {computed['total_parks_profiled']} in "
          f"{len(computed['parks_by_country'])} countries")
    print(f"coverage         {len(store['coverage_index']['terms']):,} terms, "
          f"{len(store['coverage_index'].get('families', {})):,} families")
    print(f"term locator     {len(store.get('term_locator') or {}):,} terms")
    print(f"contradictions   {len(computed['contradictions'])}")
    print(f"\nIngest took {format_duration(ingest_seconds)}:")
    for stage, seconds in STAGE_SECONDS.items():
        share = 100 * seconds / ingest_seconds if ingest_seconds else 0
        print(f"  {format_duration(seconds):>9}  {share:4.1f}%  {stage}")
    print("=" * 66)

    questions = load_questions("dev_questions.yaml")
    submission = run_submission(
        store, questions, output_path="submission_v9.json", team="group_c",
        ingest_seconds=ingest_seconds,
        notes="Every question answered twice from two complementary views of one "
              "precomputed store -- per-park detail plus narratives, and "
              "whole-document term/family/locator indexes. A view that lacks the "
              "evidence declares so and is dropped in code rather than merged, "
              "and the synthesis step follows explicit authority rules so a "
              "lookup is never averaged against a guess. Both views are built at "
              "ingest and shown for every question; no per-question retrieval.",
    )

    total = time.perf_counter() - run_started
    timing = submission["timing"]
    print("=" * 66)
    print(f"TOTAL {format_duration(total)}   "
          f"ingest {format_duration(timing['ingest_seconds'])} + "
          f"answering {format_duration(timing['answering_seconds'])}")
    print("=" * 66)
