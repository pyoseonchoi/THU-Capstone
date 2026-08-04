"""
Model H1: v14's answering machinery, with the national-parks compiler replaced.

v14 scored 80.7% on the parks book. Its answering half -- fifteen verbatim
readers plus a computed store, combined by a two-block synthesis where the
computed block cannot be overruled -- is document-agnostic and is kept exactly.
Its ingest half is not: find_park_entries, the "Park in numbers" box pairing and
the designation table all look for structures the HDR does not contain, and on
this document they return nothing. Running v14 unchanged here would silently
lose Aggregation, Superlative and Needle, which are the three categories it is
best at.

So the compiler is swapped wholesale for hdr_compile, and everything the HDR
question set can settle in Python is settled there:

    h01 h04 h05 h06   Statistical Annex Table 1, parsed to 193 ranked rows
    h02 h03           the Contents lists, counted per section
    h07 h08 h09       a BODY-ONLY coverage index

The body-only index is the one genuinely new idea. The report carries an
87-page reference section, and h07 asks which of four subjects is "never
mentioned". Three occur freely; "metaverse" occurs exactly once in the entire
file -- inside the title of a cited ITU report. Counting the whole file says it
is mentioned, which is the wrong answer. Counting pages 1-233 says it is not,
which is the right one.

Two things change for scale. The HDR is ~414k tokens against a 128k window --
3.2x, where the parks book was 1.2x -- so 34 readers are needed to hold it at
the ~6k words per reader that v13 established, against v14's 15. And with 34
chances at invention rather than 15, the anti-invention rule matters more, not
less; it is kept verbatim.

Requires:
    pip install openai pyyaml
    export BROKER_API_KEY=your_group_key

Run:
    nohup python -u pipeline_h1.py > run.log 2>&1 &
"""

import os
import re
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
from pipeline_v9 import NO_EVIDENCE, declares_no_evidence
import hdr_compile

DOCUMENT = os.environ.get("HDR_DOCUMENT", "hdr2025.txt")
QUESTIONS = os.environ.get("HDR_QUESTIONS", "hdr_questions.yaml")
DEFAULT_SUBMISSION = "submission_h3.json"      # two ahead of the pipeline number

# v13 established ~6,200 words per reader as the point where a reader
# enumerates instead of generalizing. 211k words / 6.2k -> 34.
RAW_CHUNKS = int(os.environ.get("HDR_CHUNKS", "34"))
CHUNK_OVERLAP = 300
CHUNK_WORKERS = int(os.environ.get("HDR_WORKERS", "6"))

# The one component whose input depends on the question. It selects no text and
# hides none -- see check_question_terms -- but it is the only thing here that
# is not byte-identical across questions, so it has an off switch:
#   HDR_TERM_CHECK=off python pipeline_h1.py
TERM_CHECK = os.environ.get("HDR_TERM_CHECK", "on").strip().lower() != "off"


# ------------------------------------------------------------ THE STORE

def build_store(document_text: str, checkpoint_path: str = "store_h1.json") -> dict:
    """
    Compile the document once. No LLM calls, no question consulted.

    Fast enough (a few seconds) that there is no resume path to maintain -- the
    parks pipeline needed checkpoints because its ingest called a model 60
    times; this one is pure parsing.
    """
    with timed("parse statistical annex table 1"):
        rows = hdr_compile.parse_annex_table1(document_text)
        table1 = hdr_compile.summarize_table1(rows)

    with timed("parse the contents lists"):
        contents = hdr_compile.parse_contents(document_text)

    with timed("index figure captions"):
        figures = hdr_compile.build_figure_index(document_text)

    store = {
        "document": {
            "words": len(document_text.split()),
            "pages": len(hdr_compile.split_pages(document_text)),
            "body_pages": f"1-{hdr_compile.BODY_LAST_PAGE}",
            "reference_pages": f"{hdr_compile.BODY_LAST_PAGE + 1}-end",
        },
        "computed": {
            "annex_table1": table1,
            "annex_table1_rows": rows,
            "contents": {k: {kk: vv for kk, vv in v.items() if kk != "identifiers"}
                         for k, v in contents.items()},
            "contents_identifiers": {k: v["identifiers"] for k, v in contents.items()},
            "figure_index": figures,
        },
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    return store


# ------------------------------------------------- DETERMINISTIC TERM CHECK

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "is", "are",
    "was", "were", "which", "what", "how", "who", "whom", "does", "do", "did",
    "report", "says", "say", "said", "that", "this", "these", "those", "it",
    "its", "by", "with", "as", "at", "from", "be", "been", "has", "have", "any",
    "never", "ever", "mentioned", "anywhere", "subject", "among", "each", "all",
    "give", "gives", "state", "states", "identify", "including", "include",
    "many", "much", "there", "their", "them", "his", "her", "not", "no", "one",
    "two", "three", "four", "both", "other", "than", "then", "also", "about",
}


def check_question_terms(question: str, body: str, tail: str) -> list[dict]:
    """
    Count every phrase the question itself names, in the body and in the tail.

    This does NOT select, rank or filter any part of the document: all 34
    readers still read all 34 slices for every question, and the store handed
    to the synthesis is byte-identical from question to question apart from
    this block. What it adds is a computed fact ABOUT THE QUESTION -- "the
    phrase 'universal basic income' occurs 0 times in pages 1-233" -- which no
    reader holding one thirty-fourth of the text can establish, and which the
    absence questions turn on entirely.

    Only phrases absent from the body are reported. A phrase that appears is
    not news; a phrase that does not is the answer to an absence question.
    """
    # Cut the question at every comma, conjunction and clause break first.
    # Sliding a window over the whole sentence instead produced phrases that
    # straddle the list -- "deepfakes algorithmic management" is absent from
    # the report only because it is two different list items, and reporting it
    # buried the real answer under nonsense.
    segments = re.split(r"[,;:.?()—–-]|\band\b|\bor\b|\bwhile\b|\byet\b|\bbut\b",
                        question.lower())

    phrases = set()
    for segment in segments:
        words = re.findall(r"[a-z][a-z'-]+", segment)
        for size in (1, 2, 3, 4):
            for i in range(len(words) - size + 1):
                gram = words[i:i + size]
                # A subject the report could discuss contains no function
                # words at all. This is what separates "care technologies"
                # and "universal basic income" from "ranked countries or
                # territories fall into each" -- the latter is question
                # grammar, and its absence means nothing.
                if any(w in _STOPWORDS for w in gram):
                    continue
                phrases.add(" ".join(gram))

    def absent(phrase: str) -> bool:
        """Absent only if neither the phrase nor its singular form occurs."""
        forms = {phrase}
        if phrase.endswith("s"):
            forms.add(phrase[:-1])
        if phrase.endswith("ies"):
            forms.add(phrase[:-3] + "y")
        return not any(body.count(f) for f in forms)

    findings = []
    for phrase in sorted(phrases):
        if len(phrase) < 6 or not absent(phrase):
            continue
        in_tail = tail.count(phrase)
        findings.append({
            "phrase": phrase,
            "occurrences_in_report_body": 0,
            "occurrences_in_reference_section": in_tail,
            "verdict": ("never appears in the report body; appears only inside "
                        f"cited works in the reference section ({in_tail})")
            if in_tail else "never appears anywhere in the document",
        })
    # Longest first: "universal basic income" is the finding, "basic income" is
    # the same finding stated less precisely.
    findings.sort(key=lambda f: -len(f["phrase"]))
    return findings[:24]


# ---------------------------------------------------------------- READERS

def chunk_raw_text(text: str, chunks: int = RAW_CHUNKS) -> list[dict]:
    """Verbatim slices. Every slice is read for every question; none is selected."""
    words = text.split()
    size = len(words) // chunks + 1

    slices = []
    for i in range(chunks):
        start = max(0, i * size - (CHUNK_OVERLAP if i else 0))
        end = min(len(words), (i + 1) * size + CHUNK_OVERLAP)
        slices.append({"n": i + 1, "of": chunks, "text": " ".join(words[start:end])})
        if end >= len(words):
            break
    return slices


CHUNK_PROMPT = """You are reading part {n} of {of} of the Human Development Report 2025,
"A matter of choice: People and possibilities in the age of AI", published by
UNDP. The text below is verbatim -- nothing has been summarized or removed.
Page markers like [page 57] mark where each printed page begins.

Answer the question using ONLY this text. Every name, figure, percentage and
country you write must be findable in the text below. NEVER supply a fact from
your own knowledge of this report or of AI: another reader may hold the real
evidence, and an invented detail displaces it.

You hold roughly {frac} of the report. Most parts will not contain what is
asked. If this part has nothing relevant, reply with exactly:

    {marker}

and nothing else. DO NOT GUESS.

## How to answer when you DO have something

Give up to two kinds of line.

PATTERN lines -- at most two, and ONLY if this part actually states the idea:

    PATTERN: <the claim this part makes, quoting or closely restating a
    sentence from the text>

Use these when the question asks how the report treats a subject, what it
argues repeatedly, or what it links something to. Do not write a pattern you
cannot point to a sentence for.

INSTANCE lines -- one per concrete example:

    WHERE -- the specific fact -- figure, percentage, year, country or name

Rules for instance lines:

- SAY WHERE IT COMES FROM on every line: the chapter, the box, the figure
  number ("Figure 5.5"), the table, or the page marker nearest to it.
- Copy figures, percentages, years and proper names EXACTLY as printed.
- KEEP QUALIFIERS. "about two-thirds" is not "most". "projected to" is not
  "did". "in 2023" is not "recently". The qualifier is often the whole point.
- AT MOST 12 instance lines. If this part has more, give the 12 most specific.
- Do not count anything across the whole report; you see only {frac} of it. If
  the question asks for a total, a rank or which subject is never mentioned,
  reply "{marker}" unless this part states the answer outright.

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""

INDEX_PROMPT = """Below is a store of values computed in Python over the ENTIRE Human
Development Report 2025 at ingest: every ranked row of Statistical Annex
Table 1, the Contents lists counted per section, the figure captions, and a
check of which phrases from the question appear nowhere in the report body.

Answer the question from this store alone.

The store is authoritative for anything countable: totals, counts per group,
maxima, minima, ranks, and which subjects the report never mentions. It holds
no prose and cannot supply quotations, arguments or anecdotes -- other readers
hold the text itself.

If the store does not contain what the question asks, reply with exactly:

    {marker}

and nothing else. Do not reason beyond the values given and do not estimate.

When the question asks which subject is never mentioned, answer from
question_term_check: a phrase with occurrences_in_report_body = 0 is never
mentioned in the report, even if it occurs in the reference section, because
that means it appears only inside the title of a work the report cites.

STORE:
{store}

QUESTION:
{question}

Answer with the specific values, plain text, no preamble.
"""

SYNTHESIS_PROMPT = """Write one final answer to the question, using the two blocks below.

They are NOT equal sources.

============================ HOW TO USE THEM ============================

COMPUTED VALUES were calculated in code over the ENTIRE report: every ranked
row of Statistical Annex Table 1, the Contents lists, figure captions, and
which phrases from the question appear nowhere in the report body.

    THIS BLOCK IS AUTHORITATIVE AND CANNOT BE OVERRULED. For any total, count,
    ranking, maximum, printed figure, or verdict about what the report never
    mentions, USE ITS VALUE EXACTLY -- even if several readers say otherwise.
    Readers see one slice each and cannot count, rank, or establish absence.
    If a reader's claim conflicts with this block, THE READER IS WRONG: drop
    the reader's version entirely and do not mention it.

TEXT EVIDENCE is what readers found in their own slice of the verbatim text.
Each holds about one thirty-fourth of the report. They are the only source for
specific facts, quotations, arguments, figures and examples.

    Within this block, COMBINE EVERYTHING. If one reader names three findings
    and another names two, your answer gives all five -- take the UNION, never
    a substitution. Two readers naming different things is not a conflict; it
    is two facts, and you keep both.

    PATTERN lines state what the report repeatedly argues. When the question
    asks how the report treats a subject, what it links, or how it develops an
    idea across chapters, state the pattern the readers report, then give their
    named examples and figures underneath it.

============================== HARD RULES ==============================

- NAME NOTHING THAT IS NOT IN EITHER BLOCK. If a country, figure, percentage,
  person or claim appears in no reader's lines and in no computed value, it
  does not go in your answer, however plausible it looks. This is the most
  damaging error available to you.
- Every specific detail from the text evidence must survive into your answer:
  every figure number, percentage, year, country and qualifier. Length costs
  nothing.
- Where the question names a chapter, a figure or a table, keep that label in
  your answer so the claim stays locatable.
- Give exactly ONE verdict where the question asks for one -- one absent
  subject, one highest value, one contradiction. Never offer alternatives for
  a verdict.
- Where the question asks you to state two claims that sit in tension, state
  BOTH in full, then say plainly how they are reconciled.
- Never mention the blocks, the readers, or that you were given anything.
- Never hedge, never say "it might be". Commit.

=========================== COMPUTED VALUES ===========================
{computed}

============================ TEXT EVIDENCE ============================
{evidence}

=============================== QUESTION ==============================
{question}

Final answer, plain text, no JSON, no preamble. Use as many sentences as the
details require -- being thorough is free, being brief is not. If the question
is thematic, state the pattern first and then give every named example.
"""


def answer_question(views: dict, question: str) -> dict:
    """34 verbatim readers plus the computed store, then a two-block synthesis."""
    store_view = dict(views["computed"])
    if TERM_CHECK:
        store_view["question_term_check"] = check_question_terms(
            question, views["body"], views["tail"])

    jobs = [("index", INDEX_PROMPT.format(store=_dumps(store_view),
                                          marker=NO_EVIDENCE, question=question))]
    for chunk in views["chunks"]:
        jobs.append((f"part {chunk['n']}/{chunk['of']}", CHUNK_PROMPT.format(
            n=chunk["n"], of=chunk["of"], marker=NO_EVIDENCE,
            frac=f"one {chunk['of']}th", chunk=chunk["text"], question=question)))

    def run(job):
        name, prompt = job
        try:
            return name, call_llm(prompt, model=MODEL_ANSWER,
                                  temperature=ANSWER_TEMPERATURE).strip()
        except Exception as e:
            print(f"  ! reader {name} failed: {e}")
            return name, ""

    with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
        results = list(pool.map(run, jobs))

    index_answer, raw = "", []
    for name, text in results:
        if declares_no_evidence(text):
            continue
        if name == "index":
            index_answer = text
        else:
            raw.append((name, text))

    if not index_answer and not raw:
        raise RuntimeError("no reader had evidence for this question")

    computed_block = (index_answer or
                      "(the computed store had nothing to add for this question)")
    evidence_block = "\n\n".join(f"READER {name}:\n{text}" for name, text in raw) \
        or "(no reader's slice contained anything relevant)"

    def fallback(decision):
        return {"answer": index_answer or max((t for _, t in raw), key=len),
                "computed": computed_block, "evidence": evidence_block,
                "readers": len(raw) + bool(index_answer), "decision": decision}

    try:
        final = call_llm(
            SYNTHESIS_PROMPT.format(computed=computed_block,
                                    evidence=evidence_block, question=question),
            model=MODEL_ANSWER, temperature=ANSWER_TEMPERATURE).strip()
    except Exception as e:
        print(f"  ! synthesis failed ({e}); falling back")
        return fallback("synthesis_failed")
    if not final:
        return fallback("synthesis_empty")

    return {"answer": final, "computed": computed_block,
            "evidence": evidence_block, "readers": len(raw) + bool(index_answer),
            "decision": "synthesized"}


def run_submission(store: dict, document_text: str, questions: list[dict],
                   output_path: str, team: str = "group_c", notes: str = "",
                   ingest_seconds: float = 0.0) -> dict:
    with timed("build the reader views"):
        pages = hdr_compile.split_pages(document_text)
        last = hdr_compile.BODY_LAST_PAGE
        views = {
            "computed": store["computed"],
            "chunks": chunk_raw_text(document_text),
            "body": "\n".join(pages[p] for p in sorted(pages) if p <= last).lower(),
            "tail": "\n".join(pages[p] for p in sorted(pages) if p > last).lower(),
        }
    chunks = views["chunks"]
    print(f"  computed store ~= {_estimate_tokens(_dumps(store['computed'])):,} tokens")
    print(f"  {len(chunks)} raw readers ~= "
          f"{_estimate_tokens(chunks[0]['text']):,} tokens each")

    started = time.perf_counter()
    submission = {
        "team": team, "notes": notes, "answers": [],
        "timing": {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ingest_seconds": round(ingest_seconds, 2),
            "raw_readers": len(chunks),
            "stages": {k: round(v, 2) for k, v in STAGE_SECONDS.items()},
        },
    }
    per_question, decisions = {}, {}

    for q in questions:
        qid = q["id"]
        print(f"Answering {qid}...")
        entry = {"id": qid}
        question_start = time.perf_counter()
        try:
            result = answer_question(views, q["question"])
            entry["answer"] = result["answer"]
            entry["decision"] = result["decision"]
            entry["readers_with_evidence"] = result["readers"]
            entry["evidence"] = [f"COMPUTED: {result['computed']}",
                                 f"TEXT EVIDENCE:\n{result['evidence']}"]
            decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
        except Exception as e:
            entry["answer"] = ""
            entry["error"] = str(e)
            print(f"  -> failed: {e}")

        elapsed = time.perf_counter() - question_start
        per_question[qid] = round(elapsed, 2)
        entry["seconds"] = round(elapsed, 2)
        print(f"  [{format_duration(elapsed):>9}] {qid} "
              f"({entry.get('readers_with_evidence', 0)} readers)")

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
    print(f"  answering: {format_duration(answering)} total, "
          f"{format_duration(answering / max(1, len(per_question)))} average")
    return submission


if __name__ == "__main__":
    run_started = time.perf_counter()

    with timed("read and preprocess document"):
        with open(DOCUMENT, "r", encoding="utf-8") as f:
            document_text = preprocess_document(f.read())

    store = build_store(document_text)
    ingest_seconds = time.perf_counter() - run_started
    computed = store["computed"]
    table1 = computed["annex_table1"]

    print(f"\n{'=' * 68}")
    print(f"document        {store['document']['words']:,} words, "
          f"{store['document']['pages']} pages "
          f"(body {store['document']['body_pages']})")
    print(f"annex table 1   {table1['ranked_entries_total']} ranked entries")
    for band, n in table1["ranked_entries_by_hdi_group"].items():
        print(f"                  {band:<30} {n:>4}")
    for key in ("highest_hdi_2023", "highest_life_expectancy_2023",
                "highest_gni_per_capita_2023"):
        v = table1[key]
        print(f"  {key:<30} {v['country']} = {v['value']}")
    print("contents        " + ", ".join(
        f"{k} {v['count']}" for k, v in computed["contents"].items()))
    print(f"figures indexed {len(computed['figure_index'])}")
    print(f"\nreaders per question: {RAW_CHUNKS} raw + 1 computed store, "
          f"{CHUNK_WORKERS} in flight")
    print(f"Ingest took {format_duration(ingest_seconds)}")
    print("=" * 68)

    questions = load_questions(QUESTIONS)
    submission = run_submission(
        store, document_text, questions,
        output_path=os.environ.get("SUBMISSION_FILE", DEFAULT_SUBMISSION),
        team="group_c", ingest_seconds=ingest_seconds,
        notes="Thirty-four readers hold the report's verbatim text in slices and "
              "report the patterns and concrete instances each contains; a "
              "thirty-fifth holds a store compiled at ingest in Python -- every "
              "ranked row of Statistical Annex Table 1, the Contents lists "
              "counted per section, figure captions, and a coverage index that "
              "separates the report body (pages 1-233) from the reference "
              "section, so a term occurring only inside a cited work's title is "
              "correctly reported as never discussed. The synthesis receives "
              "them as two separate blocks: the computed store is authoritative "
              "for anything countable and cannot be overruled, while the "
              "readers' evidence is combined by union for specific facts. "
              "Nothing may be named that appears in neither block. Every reader "
              "sees every slice for every question; no chunk is ever selected, "
              "ranked or filtered by the question.",
    )

    total = time.perf_counter() - run_started
    timing = submission["timing"]
    print("=" * 68)
    print(f"TOTAL {format_duration(total)}   "
          f"ingest {format_duration(timing['ingest_seconds'])} + "
          f"answering {format_duration(timing['answering_seconds'])}")
    print("=" * 68)
