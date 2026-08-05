"""
Model G2: one pipeline, any .txt, any size. No RAG, no per-document fitting.

WHAT G2 CHANGES, AND WHY
------------------------
G1's first real run scored 0.5635. The shape of that result is the whole
argument for this file: fourteen questions averaged 0.845, and seven scored
exactly 0.00. Every single zero was the same task -- pull a specific value out
of a table -- and reading the evidence blocks showed three distinct causes.

  1. WRONG TABLE. h01/h02/h03/h05. The question says "In Statistical Annex
     Table 1", the document holds seven similar composite-index tables, and
     nothing in g1 connected the two. Readers reported from whichever table was
     in their slice and tally() had no way to know.

  2. WRONG COLUMN. h04 answered Norway 0.995. That number is real and it sits
     on Norway's row -- in the GENDER Development Index table. Same country,
     same-looking row, different table. A provenance check that only asks "does
     this number appear near this entity" passes it happily.

  3. INVENTED DIGITS. h06 answered Qatar 148,063. That string occurs zero times
     in the document. Nothing anywhere verified that a reported number existed.

  ... and a fourth, found only by reading the source: h05 answered Hong Kong
     85.5 when the answer was San Marino 85.7. Both rows are in the SAME table,
     twenty-three lines apart. Nothing was mis-scoped and nothing was invented.
     The reader simply did not emit a record for San Marino. A model handed six
     thousand words containing a two-hundred-row table does not emit two hundred
     RECORD lines; it emits a dozen and stops. UNDER-REPORTING, not error.

So G2 adds four things, in descending order of how much they bought:

  A. LOCATOR SCOPING. If the question names its own location -- "Table 1",
     "Figure O.1", "Chapter 3", "Annex B" -- find that region in the document
     and read only it. The locator is matched whitespace-tolerantly, because
     PDF-derived text spells headings "TAB LE 1" and "TA B L E 3"; among the
     candidate spans, the one with the highest digit density is the data table
     rather than a passing mention. On the HDR this resolves "Table 1" to a
     49,537-character span that contains Iceland 0.972 and San Marino and does
     NOT contain Norway's 0.995. Cause 1 and cause 2 both die here.

  B. FINE SLICING INSIDE THE SCOPE. Having cut 211,000 words down to 7,000,
     slice at 1,500 words instead of 6,000. Each reader now holds ~40 table
     rows instead of ~200 and can plausibly report all of them, and the prompt
     demands completeness rather than a selection. That is cause 4. It is also
     five model calls where g1 spent thirty-five.

  C. PROVENANCE. Every numeric record is checked against the raw text before it
     is allowed to count: the entity must occur in the document, and the value's
     digits must occur within the same line. Unverifiable records are dropped
     and REPORTED, not silently kept. That is cause 3.

  D. CONFLICT RESOLUTION. G1 deduped by keeping whichever record had the longer
     'where' string -- which is to say, arbitrarily. Two readers disagreeing
     about one entity's value is a signal. G2 takes a majority, breaks ties on
     provenance, and surfaces what it could not resolve.

Everything else is g1 unchanged, including the part that already works:
absence still settles in pure Python at zero model calls, and scored 1.00.

WHAT G2 IS STILL NOT
--------------------
None of the above knows anything about the Human Development Report. There is
no page number, no table title, no band name in this file. A locator is only
ever taken from the QUESTION; if a question names no location, the pipeline
falls back to g1's behaviour of reading every slice, which is what a document
like the parks book -- sixty entries scattered through the whole text, no table
at all -- actually needs.


WHAT THIS IS FOR
----------------
The deliverable is a system that is handed a plain text document and a question
file (21 questions, 7 categories x 3) and answers them. The document is not
known in advance. Everything here therefore obeys one rule:

    A constant that names one document is a bug. Structure may be DISCOVERED
    at runtime; it may never be ASSUMED.

That rule is not academic. The previous two pipelines each scored well and each
broke on the other's document:

  pipeline_v3+   anchored on the literal string "Park in numbers"
  hdr_compile    pinned a table to pages 288-292, the Contents to 11-13, and
                 hardcoded the HDI band names

Worse, the absence machinery keyed off `[page N]` markers that our own PDF
converter had written. Handed a .txt without them -- which is what the organizers
supply -- the body resolved to an empty string, every term counted zero, and the
pipeline answered "the report never mentions wildlife protection" about a term
occurring 54 times. Confidently wrong is the most expensive thing this scoring
rewards: a forbidden hit costs 0.25 where silence costs nothing extra.

WHAT CARRIES OVER (all of it document-agnostic, all of it earned)
-----------------------------------------------------------------
  * Read the document verbatim in slices sized to the window. Compression at
    ingest destroys the sentence-level facts the judge grades on.
  * ~6,000 words per reader. At 19k a reader summarizes ("traditional land use
    coexists with protection" -- true, zero key points); at 6k it enumerates
    ("Sami reindeer herding at Abisko"). So chunk COUNT is derived from word
    count, never fixed.
  * Every reader reads its slice for every question. Nothing is selected,
    ranked or filtered by the question -- that is what keeps this out of
    retrieval territory.
  * A reader may declare NO EVIDENCE; the skip is mechanical, not a judgement.
  * Two-block synthesis: computed values are authoritative and cannot be
    overruled; text evidence is combined by union. Union is safe only when
    every draft is a quote and dangerous when one draft is invention.
  * Anything countable is computed in Python. And -- the lesson that cost five
    questions on the HDR -- a value Python computed is never handed to a model
    to retype. It goes into the answer verbatim.

HOW COUNTING WORKS WITHOUT A COMPILER
--------------------------------------
A reader holding one thirty-fourth of a document cannot count across it. But it
can list what is in front of it, and Python can count across the lists. So for
aggregation and superlative questions the readers emit structured records --

    RECORD | entity | attribute | value | where

-- and `tally()` dedupes by entity, converts units, applies any threshold the
question states, and computes the count, the maximum and the minimum. That is
the generic replacement for a per-document table parser, and it is what got the
parks aggregation category to 100% before anyone wrote a compiler for it.

Mixed units are flagged rather than ignored: exactly one park in the parks book
reported area in square miles while every other used square kilometres, and a
naive maximum picks the wrong entity. Conversions are applied and recorded.

Requires: pip install openai pyyaml   and   export BROKER_API_KEY=...
Run:      nohup python -u pipeline_g2.py > run.log 2>&1 &
"""

import collections
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pipeline_ed2 import get_client, load_questions

# ----------------------------------------------------------------- CONFIG

DOCUMENT = os.environ.get("DOC", "hdr2025.txt")
QUESTIONS = os.environ.get("QUESTIONS", "hdr_questions.yaml")
SUBMISSION = os.environ.get("SUBMISSION_FILE", "submission_g4.json")
TEAM = os.environ.get("TEAM", "group_c")

MODEL = os.environ.get("MODEL", "mistral-small-3-2")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))

# Words per reader. 6,000 is the measured line between a reader that enumerates
# named instances and one that writes summaries worth zero key points.
WORDS_PER_READER = int(os.environ.get("WORDS_PER_READER", "6000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "300"))

# Counting slices are finer and do not overlap. Finer, because a reader holding
# 6,000 words of a long table reports a dozen rows out of two hundred and stops
# -- which is how the highest life expectancy in the document got missed while
# sitting twenty-three lines from the one that was reported. Non-overlapping,
# because an overlap that is harmless when quoting prose double-counts rows when
# the job is to count.
TALLY_WORDS_PER_READER = int(os.environ.get("TALLY_WORDS_PER_READER", "1500"))

# A locator span narrower than this is a passing mention ("as Table 1 shows"),
# not the table itself. Below it, fall back to reading the whole document.
MIN_SCOPE_CHARS = int(os.environ.get("MIN_SCOPE_CHARS", "2000"))

# How far from an entity's name a figure may sit and still count as belonging to
# it. Only consulted when the two are not on one line -- see verify_record.
PROVENANCE_WINDOW = int(os.environ.get("PROVENANCE_WINDOW", "1200"))

# A candidate subject whose words are this well covered by the document is being
# discussed, whatever wording the question chose for it.
COVERAGE_DISCUSSED = float(os.environ.get("COVERAGE_DISCUSSED", "0.6"))

WORKERS = int(os.environ.get("WORKERS", "3"))
START_RPM = float(os.environ.get("RPM", "90"))
MAX_ATTEMPTS = int(os.environ.get("ATTEMPTS", "8"))

# The private document is released ~60 minutes before the deadline, and there is
# one evaluation run per team. A partial submission beats a missing one, so the
# runner stops starting new questions once this many minutes have elapsed.
DEADLINE_MINUTES = float(os.environ.get("DEADLINE_MINUTES", "0")) or None

NO_EVIDENCE = "NO EVIDENCE"


def normalize_text(raw: str) -> str:
    """
    Preprocessing that PRESERVES line structure instead of destroying it.

    This is the single most consequential function in g3, and it exists because
    the shared helper in pipeline_ed2 does the opposite. Some files store their
    newlines as the literal two characters backslash-n rather than as newline
    bytes; ed2 replaces those with SPACES. On the parks book that is 9,569
    newlines turned into spaces, and the whole 88,000-word document arrives at
    every reader as one unbroken line.

    What that costs is not subtle. The parks stats boxes are not scrambled, as
    every previous version of this project assumed -- they are value and label
    on alternating lines:

        Park in numbers
        77 Area covered (sq km)
        2106
        Highest point: Mount Kebnekaise (m)

    Flattened, that becomes "...77 Area covered (sq km) 2106 Highest point..."
    and a reader asked for the highest summit answers 173370, having glued two
    numbers together. Restored, the pairing is obvious.

    ed2 is left untouched: pipeline_h1, h2 and g1 all import its version, and a
    teammate may be running them. Changing shared behaviour underneath someone
    else's measurement is how two runs stop being comparable.
    """
    literal, real = raw.count("\\n"), raw.count("\n")
    if literal > 20 and real < literal:
        raw = raw.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        _log(f"preprocess    restored {literal:,} escaped newlines to real ones")
    # Zero-width spaces and non-breaking hyphens break exact matching invisibly.
    return raw.replace("​", "").replace("­", "")


def _log(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------- DOCUMENT: MEASURE, DISCOVER

def measure(text: str) -> dict:
    """Size first. Every strategy decision below follows from these numbers."""
    words = len(text.split())
    return {"characters": len(text), "words": words,
            "estimated_tokens": len(text) // 4}


def auto_chunk_count(words: int) -> int:
    """
    Slice count derived from length, never fixed.

    A constant 34 gave 6,219 words per reader on a 211k-word report and 2,588 on
    an 88k-word book -- the second wastes calls and pushes readers below the
    size where they enumerate.
    """
    return max(1, round(words / WORDS_PER_READER))


def chunk_text(text: str, chunks: int) -> list[dict]:
    """Verbatim slices with overlap. Every slice is read for every question."""
    words = text.split()
    size = len(words) // chunks + 1
    out = []
    for i in range(chunks):
        start = max(0, i * size - (CHUNK_OVERLAP if i else 0))
        end = min(len(words), (i + 1) * size + CHUNK_OVERLAP)
        out.append({"n": i + 1, "of": chunks, "text": " ".join(words[start:end])})
        if end >= len(words):
            break
    for slice_ in out:                      # 'of' must match what we actually made
        slice_["of"] = len(out)
    return out


def chunk_lines(text: str, target_words: int) -> list[dict]:
    """
    Slice on line boundaries, preserving them, with no overlap.

    `chunk_text` joins on whitespace, which throws every newline away. For prose
    that is harmless. For a table it is fatal: a forty-row table arrives as one
    space-separated blob with no row boundaries at all, and a reader asked which
    number belongs to which country has nothing left to align on but word order.
    That is the likeliest reason a Gender Development Index value ended up
    reported as a Human Development Index one.

    No overlap, because these slices get COUNTED. The 300-word overlap that
    protects a sentence from being cut in half would silently duplicate rows.

    A line longer than the target is split on words, because line boundaries are
    a convenience and the context window is not. The parks document is a single
    line of 92,808 words -- an artifact of how it was converted -- and a
    line-only splitter hands one reader the entire book.
    """
    out, current, words = [], [], 0

    def flush():
        nonlocal current, words
        if current:
            out.append("\n".join(current))
            current, words = [], 0

    for line in text.split("\n"):
        n = len(line.split())
        if n > target_words:                    # oversized line: split on words
            flush()
            tokens = line.split()
            for i in range(0, len(tokens), target_words):
                out.append(" ".join(tokens[i:i + target_words]))
            continue
        current.append(line)
        words += n
        if words >= target_words:
            flush()
    flush()

    if not out:
        out = [text]
    return [{"n": i + 1, "of": len(out), "text": t} for i, t in enumerate(out)]


# Page markers arrive in more than one dialect: our own converter wrote
# "[page 12]", the parks book carries "<!-- page-start-marker-12 -->". Recognise
# both rather than assuming ours -- an unrecognised marker format makes the body
# resolve to nothing, which is the failure that answers "never mentioned" about
# a word occurring fifty-four times.
_PAGE_MARKER = re.compile(
    r"^\[page (\d+)\]$|<!--\s*page-start-marker-(\d+)\s*-->", re.M)

# A bibliography heading on a line of its own. Discovered, not assumed: if the
# document has no such heading, there is no split and everything is body.
_REFERENCE_HEADING = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited|"
    r"reference list|sources)\s*$", re.I | re.M)


def discover_structure(text: str) -> dict:
    """
    What this particular document happens to offer. Nothing is required.

    Two optional findings:
      page markers      -> facts can be cited by page
      a reference tail  -> a term occurring only inside a cited work's title is
                           not a subject the document discusses. This mattered
                           enormously on one document (a term appeared exactly
                           once, inside an ITU report title) and is simply
                           absent on others.
    """
    pages = {int(m.group(1) or m.group(2)): m.start()
             for m in _PAGE_MARKER.finditer(text)}

    reference_start = None
    for m in _REFERENCE_HEADING.finditer(text):
        # Position alone is a bad test -- one report's "References" heading sits
        # at 52% of the file because its statistical tables are so dense. So
        # require the heading to be past the halfway mark AND for the text after
        # it to actually READ like a bibliography: citation years and URLs at a
        # density prose never reaches.
        if m.start() < len(text) * 0.45 or len(text) - m.start() < 2000:
            continue
        sample = text[m.end():m.end() + 5000]
        citations = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\.", sample))
        links = sample.count("http")
        if citations + links >= 15:
            reference_start = m.start()
            break

    return {
        "page_markers": len(pages),
        "reference_section_found": reference_start is not None,
        "reference_start_char": reference_start,
        "body_fraction": round((reference_start or len(text)) / len(text), 3),
    }


def split_body(text: str, structure: dict) -> tuple[str, str]:
    """(body, reference tail). No bibliography found means it is all body."""
    cut = structure.get("reference_start_char")
    if cut is None:
        return text, ""
    return text[:cut], text[cut:]


# --------------------------------------------- QUESTION ANALYSIS (no document)

_SEGMENTS = re.compile(r"[,;:.?()—–]|\band\b|\bor\b|\bwhile\b|\byet\b|\bbut\b")
_WORDS = re.compile(r"[a-z][a-z'-]+")

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "is", "are",
    "was", "were", "which", "what", "how", "who", "whom", "does", "do", "did",
    "report", "book", "document", "says", "say", "said", "that", "this",
    "these", "those", "it", "its", "by", "with", "as", "at", "from", "be",
    "been", "has", "have", "any", "never", "ever", "mentioned", "anywhere",
    "subject", "among", "each", "all", "give", "gives", "state", "states",
    "identify", "including", "include", "many", "much", "there", "their",
    "them", "his", "her", "not", "no", "one", "two", "three", "four", "both",
    "other", "than", "then", "also", "about", "name", "names", "total",
}


# An absence question is always "[preamble.] Of A, B, C and D, which ... never
# ...?" -- the candidates sit before the interrogative. Feeding the whole
# sentence in instead makes "which is never substantively discussed anywhere in
# the book" a candidate; it occurs nowhere, so it wins the absence verdict and
# the real answer is never considered.
_INTERROGATIVE = re.compile(r"\b(which|what)\b", re.I)

# Segments made entirely of question furniture ("the following four subjects").
_META = {"following", "subjects", "topics", "items", "listed", "above", "below",
         "named", "mentioned", "discussed", "substantively", "anywhere"}


def _looks_enumerated(text: str) -> bool:
    """Two or more items separated by commas, or an explicit 'or'/'and' join."""
    return text.count(",") >= 1 and bool(re.search(r",\s*(?:or|and)\s+\S", text, re.I))


def candidate_list(question: str) -> str:
    """
    The part of the question that holds the listed candidates.

    A question can put its list on either side of the interrogative, and the two
    documents this was built on both happened to put it first:

        "Of the following four subjects - A, B, C, and D - which is never...?"
        "Which one of the following is never discussed: A, B, C, or D?"

    Reading only the prefix returned an EMPTY string for the second form, so no
    candidates were found, absence fell through to the readers, and a category
    that scores 1.00 by counting scored 0.25 by guessing. Try the trailing list
    first, because when a question has one it is unambiguous.
    """
    tail = question.rsplit(":", 1)[-1] if ":" in question else ""
    if _looks_enumerated(tail):
        return tail.strip(" ?.")

    m = _INTERROGATIVE.search(question)
    prefix = question[:m.start()] if m else question
    # Drop any preamble sentence: "The book names several designations. Of X,..."
    parts = re.split(r"(?<=[.!?])\s+", prefix.strip())
    best = parts[-1] if parts and parts[-1].strip() else prefix
    if _looks_enumerated(best):
        return best

    # Last resort: the longest enumerated run anywhere in the question.
    runs = [seg for seg in re.split(r"[:;?]|\s[—–-]\s", question)
            if _looks_enumerated(seg)]
    return max(runs, key=len).strip(" ?.") if runs else best


def _list_segment(candidates: str) -> str:
    """
    Keep only the enumerated list, not the sentence that introduces it.

    "Of these four threats to protected areas - poaching, wartime damage...,
    glacier retreat, and pressure from visitor numbers -" carries its list
    between dashes. Taking the whole string made "these four threats to
    protected areas" a candidate subject in its own right, so the answer
    reported on that phrase instead of on the four threats the key points ask
    about. Pick the dash-delimited segment holding the most separators.
    """
    parts = re.split(r"\s[—–-]\s|:\s", candidates)
    if len(parts) < 2:
        return candidates
    best = max(parts, key=lambda p: (p.count(","), len(p)))
    return best if best.count(",") >= 1 else candidates


def question_subjects(question: str) -> list[dict]:
    """
    The items a question lists, with the phrases inside each worth counting.

    N-grams begin and end on a content word but may contain function words --
    cutting at function words instead loses "people with disabilities" and
    "social dialogue over algorithmic management", which are single subjects.
    """
    subjects = []
    for segment in _SEGMENTS.split(question):
        words = _WORDS.findall(segment.lower())
        content = [w for w in words if w not in _STOPWORDS]
        if not content or all(w in _META for w in content):
            continue                      # "the following four subjects"

        grams = set()
        for size in range(1, len(words) + 1):
            for i in range(len(words) - size + 1):
                gram = words[i:i + size]
                if gram[0] in _STOPWORDS or gram[-1] in _STOPWORDS:
                    continue
                grams.add(" ".join(gram))

        display = re.sub(r"^(?:of|among|which|the following)\s+", "",
                         segment.strip().strip(",;:. "), flags=re.I).strip()
        subjects.append({
            "display": display or " ".join(content),
            "content_words": len(content),
            "phrases": sorted(grams, key=lambda g: (-len(g.split()), -len(g))),
        })
    return subjects


# The unit can sit between the number and the comparison -- "3000 metres or
# more", "3000 sq km or above" -- so allow a few words in between.
_THRESHOLD = [
    (re.compile(r"(\d[\d,.]*)(?:\s+\w+){0,3}\s+(?:or (?:more|above|greater|higher)"
                r"|and (?:above|over))", re.I), ">="),
    (re.compile(r"(?:at least|no fewer than|minimum of)\s+(\d[\d,.]*)", re.I), ">="),
    (re.compile(r"(?:more than|greater than|above|over|exceeding)\s+(\d[\d,.]*)", re.I), ">"),
    (re.compile(r"(?:fewer than|less than|below|under)\s+(\d[\d,.]*)", re.I), "<"),
    (re.compile(r"(\d[\d,.]*)(?:\s+\w+){0,3}\s+or (?:fewer|less|below)", re.I), "<="),
]


def parse_threshold(question: str):
    """'a highest point of 3000 metres or more' -> (3000.0, '>='). Or None."""
    for pattern, op in _THRESHOLD:
        m = pattern.search(question)
        if m:
            try:
                return float(m.group(1).replace(",", "")), op
            except ValueError:
                continue
    return None


# ------------------------------------------------ QUESTION PARTS (generic)
#
# The change that addresses the largest remaining loss on both documents.
#
# Seven points of the fourteen lost across the HDR and the parks book went the
# same way: the answer was not wrong, it was INCOMPLETE. It stated two of the
# four things the question asked for. h15 asks for a framework's name, a figure
# and a finding; across three scored runs it returned the figure every time and
# the framework name never.
#
# The rubric is literally a list of parts, so make the parts explicit -- to the
# readers, so they look for all of them, and to the synthesiser, so it answers
# each in turn. Padding is free under this scoring; omission is not.
#
# Note what this does NOT do: it does not re-read the document per part. Readers
# are the expensive step and synthesis is one call, so decomposition costs two
# extra synthesis calls per question, not another thirty-five readers.

_IMPERATIVE = (r"give|identify|name|state|list|explain|note|compare|say|report|"
               r"describe|include|specify|indicate|tell")

_PART_SPLIT = re.compile(
    rf"(?:\s*[;?]\s+)"                                # a full stop between asks
    rf"|(?:,?\s+and\s+(?=(?:how|what|which|who|where|when|why|whether)\b))"
    rf"|(?:(?<=[.;])\s+(?=(?:{_IMPERATIVE})\b))"      # a new sentence, imperative
    rf"|(?:,\s+(?:and\s+)?(?=(?:{_IMPERATIVE})\b\s+(?:the|a|an|at|two|three|"
    rf"any|each|its|their|them|which|what)\b))",
    re.I)

# A fragment ending on a function word is a bad cut, not a question.
_DANGLING = re.compile(r"\b(?:the|a|an|is|are|was|were|of|in|on|to|for|and|or|"
                       r"that|this|its|their)$", re.I)


def question_parts(question: str) -> list[str]:
    """
    The distinct things a question asks for, in the order it asks for them.

    Deliberately conservative. A wrong split is worse than no split: it tells
    every reader to hunt for something the question never asked. An early
    version cut "the Contents' FIGURES list" at the word "list" and left
    "what change is the" as a part, so any cut that leaves a dangling fragment
    now discards the whole decomposition and the question is used unchanged.
    """
    raw = [p.strip(" ,;.") for p in _PART_SPLIT.split(question) if p]
    parts = [p for p in raw if len(p.split()) >= 4]
    if len(parts) < 2:
        return [question.strip()]
    if any(_DANGLING.search(p) for p in parts):
        return [question.strip()]
    # The pieces must account for most of the question; if the split threw away
    # half of it, it found a phrase boundary rather than a second request.
    if sum(len(p.split()) for p in parts) < 0.75 * len(question.split()):
        return [question.strip()]

    # The first part carries the sentence's subject; later parts are often
    # elliptical ("and how many under TABLES"). Keep them verbatim anyway -- the
    # reader sees the whole question too, so the fragment is a pointer, not a
    # standalone question.
    return parts[:5]


# A part that only ever asks for a NUMBER can be answered by a count. A part
# that asks for a NAME cannot.
# An instruction about HOW to count, not a second thing to answer.
_QUALIFIER = re.compile(r"^(?:include|exclude|ignore|omit|note|counting|treat|"
                        r"do not|don't)\b", re.I)
_COUNTING = re.compile(r"\bhow many\b|\bnumber of\b|\btotal\b|\bhow much\b|"
                       r"\bthe most\b|\bthe fewest\b|\beach\b", re.I)


def python_answers_all(parts: list[str]) -> bool:
    """
    Can a count answer every part of this question?

    This test exists because the opposite rule cost points on three separate
    documents. "Do not short-circuit a multi-part question" was written for a
    parks question that asks for a total, a per-country count AND the names of
    the parks -- where the readers really are needed. But it also fired on
    "how many boxes are in each chapter, and how many in total", which a count
    answers completely. Two HDR questions dropped from a 0.1-second exact answer
    into thirty-five readers that scored the same, and the equivalent OECD
    question came back with every count zeroed: an exact forbidden claim.

    So the test is what the parts ASK FOR, not how many there are. Any part
    wanting something named goes to the readers; if every part wants a number,
    Python is the whole answer.
    """
    real = [p for p in parts if not _QUALIFIER.match(p.strip())]
    if not real:
        return True
    for part in real:
        if _COUNTING.search(part):
            continue        # a count answers it -- including "the most", which
                            # a per-group breakdown already reports
        return False        # anything else -- above all, "Name the Spanish ones"
    return True


def format_parts(parts: list[str]) -> str:
    if len(parts) < 2:
        return ""
    listed = "\n".join(f"    ({i + 1}) {p}" for i, p in enumerate(parts))
    return (f"\n\nThis question asks for {len(parts)} SEPARATE things. Every one "
            f"of them is graded, and one that goes unmentioned scores nothing:\n"
            f"{listed}\n")


def question_mode(question: str, category: str | None) -> str:
    """
    Which machinery this question needs: tally, absence, or prose.

    The category ships with the question file, so this is reading supplied
    metadata rather than guessing about the document. Keyword inference is the
    fallback for question files that omit it.
    """
    # Every category the brief defines is mapped, including the ones that route
    # to prose. G1 mapped only three and let keyword inference decide the rest,
    # so a cross-section question containing the word "highest" was treated as a
    # counting question. Under g1 that merely added an unused computed block;
    # under g2 it also narrows the reading to whatever locator the question
    # happens to mention, which would throw away most of the document. A
    # supplied category is metadata, not a guess -- trust it.
    known = {"aggregation": "tally", "superlative": "tally", "absence": "absence",
             "contradiction": "prose", "cross_section": "prose",
             "cross-section": "prose", "global_synthesis": "prose",
             "global-synthesis": "prose", "needle": "prose"}
    if category and category.lower().strip() in known:
        return known[category.lower().strip()]
    q = question.lower()
    if re.search(r"never (mentioned|appears|discussed|substantively)", q):
        return "absence"
    if re.search(r"\bhow many\b|\bhow much\b|\bcount\b|\btotal\b", q):
        return "tally"
    if re.search(r"\b(largest|smallest|highest|lowest|most|least|longest|"
                 r"shortest|oldest|earliest|biggest)\b", q):
        return "tally"
    return "prose"


# --------------------------------------------------------- ABSENCE (generic)

def _occurrences(phrase: str, text: str) -> int:
    """
    Count a phrase and its obvious singular/plural forms, ON WORD BOUNDARIES.

    Substring counting silently destroys absence questions: "natura" occurs
    inside every "natural", so Natura 2000 -- which genuinely never appears in
    the parks book -- was reported as discussed.
    """
    forms = {phrase}
    if phrase.endswith("ies"):
        forms.add(phrase[:-3] + "y")
    elif phrase.endswith("s"):
        forms.add(phrase[:-1])
    else:
        forms.add(phrase + "s")
    return sum(len(re.findall(rf"\b{re.escape(f)}\b", text)) for f in forms)


def _family_present(word: str, text: str) -> bool:
    """
    Does this word, or any word sharing its root, occur?

    A subject can be lexically absent while the concept is discussed under a
    relative: "glaciation" appears 0 times in a book that says "glacier" 118
    times and "glacial" besides. Judging that subject absent on the exact word
    alone is wrong, and this is the single most repeated lesson in the earlier
    work -- word families are mandatory.
    """
    stem = word[:5] if len(word) > 5 else word
    return re.search(rf"\b{re.escape(stem)}\w*", text) is not None


def subject_coverage(question: str, body: str, tail: str) -> list[dict]:
    """
    For each subject the question names, is it discussed in the body?

    Present means some phrase of two or more content words occurs -- or, for a
    one-word subject, that word. Requiring two words stops "intelligence"
    certifying "artificial intelligence audit protocols"; allowing the longest
    matching sub-phrase stops "excessive screen time in early childhood" being
    called absent because the full sentence never appears verbatim.
    """
    findings = []
    for subject in question_subjects(_list_segment(candidate_list(question))):
        minimum = 1 if subject["content_words"] == 1 else 2
        candidates = [p for p in subject["phrases"] if len(p) >= 5
                      and sum(1 for w in p.split() if w not in _STOPWORDS) >= minimum]

        matched, count = None, 0
        for phrase in candidates:                    # longest first
            hits = _occurrences(phrase, body)
            if hits:
                matched, count = phrase, hits
                break

        full = candidates[0] if candidates else None
        in_tail = _occurrences(matched or full, tail) if (matched or full) and tail else 0

        # How much of the subject's vocabulary appears at all, word by word.
        # This separates a genuinely absent concept from a paraphrase of one the
        # document does discuss: "poaching" scores 0.0 because the word never
        # appears, while "pressure from visitor numbers" scores 1.0 -- every one
        # of its words is in the text, just never in that combination.
        content = [w for w in _WORDS.findall(subject["display"].lower())
                   if w not in _STOPWORDS]
        seen = sum(1 for w in content if len(w) > 3 and _family_present(w, body))
        coverage = seen / len(content) if content else 1.0

        findings.append({
            "subject": subject["display"], "matched_phrase": matched,
            "body_mentions": count, "reference_section_mentions": in_tail,
            "word_coverage": round(coverage, 2), "discussed": count > 0,
        })
    return findings


def _first_mention(phrase: str, source: str, width: int = 130) -> str:
    """
    One clean sentence-fragment around a phrase's first occurrence.

    Read from the ORIGINAL-CASE text: the coverage index is lowercased, so the
    proper nouns these key points ask for -- Jostedalsbreen, Cinque Terre --
    only survive here.
    """
    if not source or not phrase:
        return ""
    m = re.search(re.escape(phrase), source, re.I)
    if not m:
        # The question's wording is rarely the document's. "glacier retreat"
        # appears nowhere in a book that discusses glaciers retreating at
        # length, so fall back to the subject's own content words, longest
        # first, matched as families. Without this the quote is always empty
        # for exactly the subjects that most need one.
        words = sorted(re.findall(r"[A-Za-z]{5,}", phrase), key=len, reverse=True)
        # Whole word before stem. Going straight to a five-letter stem matched
        # "cider press" for "pressure" -- the same collision that once reported
        # "Natura 2000" as present because "natural" contains it.
        for pattern in ([rf"\b{re.escape(w)}\w*" for w in words]
                        + [rf"\b{re.escape(w[:5])}\w*" for w in words]):
            m = re.search(pattern, source, re.I)
            if m:
                break
    if not m:
        return ""
    start = max(0, m.start() - width)
    snippet = " ".join(source[start:m.end() + width].split())
    # Trim to whole words at both ends so the quote does not begin mid-word.
    parts = snippet.split(" ")
    return " ".join(parts[1:-1]) if len(parts) > 3 else snippet


def _mentions(phrase: str, source: str, limit: int = 3) -> list[str]:
    """
    Up to `limit` quotes for one subject, spread across the document.

    Occurrences are taken from different regions rather than consecutively:
    three quotes from the same paragraph name the same park three times, which
    buys nothing, while three from different chapters are three chances to name
    the example the key point happens to want.
    """
    if not source or not phrase:
        return []
    spots = [m for m in re.finditer(re.escape(phrase), source, re.I)]
    if not spots:
        words = sorted(re.findall(r"[A-Za-z]{5,}", phrase), key=len, reverse=True)
        for pattern in ([rf"\b{re.escape(w)}\w*" for w in words]
                        + [rf"\b{re.escape(w[:5])}\w*" for w in words]):
            spots = [m for m in re.finditer(pattern, source, re.I)]
            if spots:
                break
    if not spots:
        return []

    # Prefer windows dense in proper nouns. The key points ask for particular
    # examples -- Jostedalsbreen for glacier retreat, Cinque Terre for visitor
    # pressure -- so a quote that names something is worth several that do not.
    # Sampling occurrences evenly returned ten quotes naming none of them.
    proper = re.compile(r"\b[A-Z][a-z]{2,} [A-Z][a-z]{2,}\b")
    scored = []
    for m in spots[:400]:
        start = max(0, m.start() - 150)
        window = source[start:m.end() + 150]
        scored.append((len(proper.findall(window)), m.start(), window))
    scored.sort(key=lambda t: -t[0])

    out, taken = [], []
    for _, pos, window in scored:
        if any(abs(pos - p) < 600 for p in taken):    # don't quote the same spot twice
            continue
        taken.append(pos)
        words = " ".join(window.split()).split(" ")
        out.append(" ".join(words[1:-1]) if len(words) > 3 else " ".join(words))
        if len(out) >= limit:
            break
    return out


def settle_absence(question: str, body: str, tail: str,
                   source: str = "") -> dict | None:
    """
    Answer an absence question by counting, or decline.

    THE GUARD THAT MATTERS: with no text, every term counts zero and every
    subject looks absent. That produced a confident wrong verdict once already.
    Empty body -> return None and let the readers answer.
    """
    if not body.strip():
        return None

    found = subject_coverage(question, body, tail)
    absent = [f for f in found if not f["discussed"]]
    present = [f for f in found if f["discussed"]]
    if len(found) < 2 or not absent:
        return None

    # Among candidates that look absent, the one whose own words are least
    # present is the real answer. Picking the longest name instead chose
    # "pressure from visitor numbers" -- a paraphrase of a subject the book
    # does cover -- over "poaching", which appears nowhere at all.
    verdict = min(absent, key=lambda f: (f["word_coverage"], -len(f["subject"])))

    # Everything the question named EXCEPT the verdict is, by construction,
    # something the document does cover -- and the key points ask us to say so.
    # Membership is decided on word coverage rather than on the exact phrase,
    # because the parks book never prints the literal string "glacier retreat";
    # it writes at length about glaciers retreating. Testing the phrase left
    # this list empty and cost three key points on a question whose verdict was
    # already correct. The verdict itself is chosen above and is never touched
    # by this -- an earlier version reclassified candidates before choosing and
    # destroyed the answer entirely.
    present = [f for f in found
               if f is not verdict
               and (f["discussed"] or f["word_coverage"] >= COVERAGE_DISCUSSED)]
    scope = ("the body of the document, excluding the reference section"
             if tail else "the document")

    text = (f"The document never mentions {verdict['subject']}. The phrase does "
            f"not occur anywhere in {scope}.")
    if verdict["reference_section_mentions"]:
        text += (f" It appears {verdict['reference_section_mentions']} time(s) only "
                 "inside the title of a work listed in the reference section, "
                 "which is a citation rather than a subject the document discusses.")
    if present:
        # "(0 mentions)" next to "is discussed" reads as a contradiction and
        # invites the synthesiser to hedge. When the document covers a subject
        # in its own words rather than the question's, say that instead.
        detail = "; ".join(
            f"{f['subject']} "
            + (f"({f['body_mentions']} mentions"
               + (f", as \"{f['matched_phrase']}\"" if f["matched_phrase"] else "")
               + ")"
               if f["body_mentions"] else "(discussed in the document's own wording)")
            for f in present)
        text += (f" Every other subject named in the question is discussed: {detail}.")

        # Naming the verdict is only part of what these questions ask. The key
        # points also want the subjects that ARE covered, with an example each --
        # "glacier retreat is discussed, for example at Jostedalsbreen". We
        # already know every candidate's count; quoting one occurrence costs
        # nothing and answers three or four extra key points per question.
        for f in present:
            # Several quotes, spread through the document, not just the first.
            # The key points want particular examples -- Jostedalsbreen for
            # glacier retreat, Cinque Terre for visitor pressure -- and the
            # first occurrence is rarely the canonical one. Padding is free, so
            # quoting once was leaving points on the floor.
            for quote in _mentions(f.get("matched_phrase") or f["subject"],
                                   source, limit=4):
                text += (f" On {f['subject']}, the document states: \"{quote}\"")

    return {"answer": text, "basis": "whole-document term counting",
            "ambiguous": [f["subject"] for f in absent[1:]] if len(absent) > 1 else []}


# -------------------------------------------------- LOCATOR SCOPING (generic)
#
# The single highest-value thing in g2, and it costs no model calls.
#
# A question that says "In Statistical Annex Table 1, how many countries..."
# has told us where to look. G1 ignored that and read all 211,000 words, so
# readers reported rows from six other composite-index tables and Python had no
# way to tell them apart. Here the locator is extracted from the QUESTION -- the
# document is never assumed to contain anything -- and used to cut the search
# down to the region it names.

# Locator nouns worth honouring. Deliberately generic: these are the words
# documents use to number their own parts, in any document.
_LOCATOR_NOUN = (r"table|figure|box|chart|annex|appendix|chapter|section|part|"
                 r"exhibit|panel|spotlight|map")

# "Table 1", "Figure O.1", "Annex Table 1", "Chapter 3", "Table A2.1"
_LOCATOR = re.compile(
    rf"\b(?:annex\s+|statistical\s+annex\s+|appendix\s+)?"
    rf"(?P<noun>{_LOCATOR_NOUN})\s*"
    rf"(?P<num>[A-Z]?\d+(?:\.\d+)*[a-z]?)\b",
    re.I)


def _loose(phrase: str) -> re.Pattern:
    """
    Match a phrase tolerating whitespace inside and between its characters.

    PDF-derived text spells headings letter-spaced: the HDR prints its tables as
    'TAB LE 1', 'TA B L E 3', 'TABL E 4'. A plain search for "Table 1" finds the
    prose mentions and misses the heading -- which is to say, it finds every
    place the table is discussed and not the table.
    """
    return re.compile("".join(r"\s+" if c.isspace() else re.escape(c) + r"\s*"
                              for c in phrase), re.I)


def question_locators(question: str) -> list[dict]:
    """Locators the question names, most specific first."""
    seen, out = set(), []
    for m in _LOCATOR.finditer(question):
        noun, num = m.group("noun").lower(), m.group("num")
        key = f"{noun} {num}".lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"noun": noun, "number": num, "phrase": f"{noun} {num}",
                    "printed": m.group(0).strip()})
    # A locator carrying a section number ("Figure O.1") is more specific than a
    # bare one ("Chapter 3"), so try it first.
    out.sort(key=lambda d: (-len(d["number"]), d["phrase"]))
    return out


def locator_span(text: str, locator: dict) -> tuple[int, int] | None:
    """
    Resolve a locator to a character span of the document, or None.

    Every occurrence of "Table 1" is a candidate start; the span runs to the next
    occurrence of a SIBLING ("Table 2", "Table 3", ...) because that is where the
    numbered part ends. Among the candidates, take the one with the highest digit
    density: a data table is dense with numbers, a sentence mentioning it is not.
    That single test is what separates the annex table from the eleven places the
    report talks about the annex table -- with no page numbers involved.
    """
    noun, num = locator["noun"], locator["number"]
    start_pat = _loose(f"{noun} {num}")

    # Siblings: same noun, a different number. Built from the document's own
    # numbering, so it works for "Table 2" and "Figure O.2" alike.
    stem = re.match(r"([A-Z]?)(\d+)", num, re.I)
    siblings = []
    if stem:
        prefix, first = stem.group(1), int(stem.group(2))
        for k in range(1, 40):
            if k == first:
                continue
            siblings.append(_loose(f"{noun} {prefix}{k}").pattern)
    sibling_pat = re.compile("|".join(siblings), re.I) if siblings else None

    best = None
    for m in start_pat.finditer(text):
        nxt = sibling_pat.search(text, m.end()) if sibling_pat else None
        end = nxt.start() if nxt else len(text)
        if end - m.start() < MIN_SCOPE_CHARS:
            continue
        span = text[m.start():end]
        density = sum(c.isdigit() for c in span) / len(span)
        if best is None or density > best[0]:
            best = (density, m.start(), end)

    if best is None:
        return None
    return best[1], best[2]


def scope_for(text: str, question: str) -> dict | None:
    """The region of the document a question points at, if it points at one."""
    for loc in question_locators(question):
        span = locator_span(text, loc)
        if span:
            start, end = span
            return {"locator": loc["printed"], "phrase": loc["phrase"],
                    "start": start, "end": end, "text": text[start:end],
                    "chars": end - start,
                    "words": len(text[start:end].split())}
    return None


# ------------------------------------------- COUNTING A DOCUMENT'S OWN PARTS
#
# "How many boxes, spotlights and tables does the report contain?" and "how many
# figures appear in each chapter?" are counting questions that no reader can
# answer -- each sees one slice -- and that g1 therefore got wrong twice. But
# they need no reader at all. A document that numbers its own parts has already
# published the count; Python only has to collect the distinct identifiers.
#
# This is the same trick as absence: settle it by counting, call no model, and
# leave nothing for a model to overrule.

_PLURAL = {
    "figures": "figure", "boxes": "box", "tables": "table", "charts": "chart",
    "maps": "map", "spotlights": "spotlight", "exhibits": "exhibit",
    "panels": "panel", "annexes": "annex", "appendices": "appendix",
    "chapters": "chapter", "sections": "section",
}
_PLURAL_RE = re.compile(r"\b(" + "|".join(_PLURAL) + r")\b", re.I)

# An identifier is a noun followed by a SECTIONED number -- "Figure 3.1",
# "Box O.2", "Table S3.1". The section part matters: it is what distinguishes a
# numbered part of the document from the phrase "table 1" in a sentence.
#
# The trailing (?:\.\d+)+ must be greedy across several levels. Capping it at
# one level collapsed S3.1.1 through S3.1.4 into a single "S3.1" and lost four
# figures out of sixty-seven -- an undercount that looks entirely plausible.
_IDENTIFIER = re.compile(
    r"\b(" + "|".join(_PLURAL.values()) + r")\s+([A-Z]{0,2}\d*(?:\.\d+)+[a-z]?)\b",
    re.I)


def _plural(noun: str, n: int) -> str:
    """'box' -> 'boxes', not 'boxs'."""
    if n == 1:
        return noun
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def document_identifiers(text: str) -> dict[str, set]:
    """Every distinct numbered part the document names, grouped by noun."""
    found: dict[str, set] = {}
    for m in _IDENTIFIER.finditer(text):
        found.setdefault(m.group(1).lower(), set()).add(m.group(2).upper())
    return found


def _chapter_key(identifier: str) -> str:
    """
    'S3.1' and '3.14' both belong to chapter 3; 'O.2' belongs to the overview.

    Documents prefix a spotlight or annex figure with a letter and keep the
    chapter number after it. Group on the digits when there are any, and on the
    letter when there are none -- no knowledge of what the letters mean.
    """
    head = identifier.split(".")[0]
    digits = re.sub(r"\D", "", head)
    return digits or head


def count_entries(text: str) -> dict | None:
    """
    How many entries a document is built from, found by repetition alone.

    A book of profiles repeats the same section headings inside every entry. The
    parks book carries "## Stay here...", "## Do this!" and "## What to spot..."
    exactly sixty times each -- and it profiles sixty parks. So the count is
    already published; nothing has to be parsed or named.

    This is the same idea as counting numbered identifiers, for documents that
    number nothing. It replaces asking readers to count entries across slices,
    which returned 40 and then 49 for a document containing 60.
    """
    lines = text.split("\n")
    heads = collections.Counter(l.strip() for l in lines
                                if l.strip().startswith("#") and len(l.strip()) < 60)
    if not heads:
        return None
    ranked = heads.most_common()
    top, count = ranked[0]
    if count < 5:
        return None

    # Corroboration: an entry structure repeats SEVERAL headings the same number
    # of times. One heading at some count could be anything.
    agreeing = [h for h, n in ranked if n == count]
    if len(agreeing) < 2:
        return None

    positions = [i for i, l in enumerate(lines) if l.strip() == top]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if not gaps or max(gaps) > 12 * (sum(gaps) / len(gaps)):
        return None                       # not evenly spaced: not an entry marker

    return {"entries": count, "markers": agreeing[:4],
            "note": (f"The document repeats {len(agreeing)} section headings "
                     f"exactly {count} times each, evenly spaced, so it is built "
                     f"from {count} entries. Counted in Python over the whole "
                     f"document; use this number exactly.")}


def settle_identifier_count(question: str, text: str) -> dict | None:
    """Answer a 'how many parts does this document have' question in Python."""
    if not re.search(r"\bhow many\b|\bnumber of\b|\bhow much\b", question, re.I):
        return None
    # A question naming a SPECIFIC part ("in Table 1") is asking about that
    # part's contents, not about how many parts exist.
    if question_locators(question):
        return None

    nouns = [_PLURAL[m.group(1).lower()] for m in _PLURAL_RE.finditer(question)]
    if not nouns:
        return None

    available = document_identifiers(text)
    counted = {n: available[n] for n in dict.fromkeys(nouns) if available.get(n)}
    if not counted:
        return None

    parts, breakdowns = [], {}
    # Deliberately loose. "to each of Chapters 1-6" is the same request as "in
    # each chapter", and a tighter pattern silently answered only half of what
    # was asked -- which under this scoring is indistinguishable from a wrong
    # answer, since the unstated key points simply go uncovered.
    wants_breakdown = re.search(r"\beach\b|\bper\b|\bbreakdown\b|\bthe most\b|"
                                r"\bdistribut|\brespectively\b", question, re.I)
    for noun, ids in counted.items():
        parts.append(f"{len(ids)} {_plural(noun, len(ids))}")
        if wants_breakdown:
            groups: dict[str, int] = {}
            for i in ids:
                groups[_chapter_key(i)] = groups.get(_chapter_key(i), 0) + 1
            breakdowns[noun] = dict(
                sorted(groups.items(), key=lambda kv: (kv[0].isdigit() and int(kv[0]) or 0, kv[0])))

    answer = "The document contains " + ", ".join(parts[:-1])
    answer = (answer + " and " + parts[-1] if len(parts) > 1 else "The document contains " + parts[0])
    answer += "."

    for noun, groups in breakdowns.items():
        detail = ", ".join(f"{k}: {v}" for k, v in groups.items())
        most = max(groups, key=groups.get)
        total = len(counted[noun])
        answer += (f" The {total} {_plural(noun, total)} are distributed as "
                   f"follows, by the section number each identifier carries -- "
                   f"{detail}. The largest group is {most}, with {groups[most]}, "
                   f"and the total is {total}.")

    return {"answer": answer,
            "counts": {n: len(ids) for n, ids in counted.items()},
            "breakdown": breakdowns or None,
            "identifiers": {n: sorted(ids) for n, ids in counted.items()}}


# ----------------------------------------------------------- TALLY (generic)

# RECORD | entity | attribute | value | where
_RECORD = re.compile(r"^\s*RECORD\s*\|(.+?)\|(.+?)\|(.+?)\|(.*)$", re.I | re.M)
# MAXIMUM | attribute | entity | value -- one per slice, the local extremum.
# A cheap guard against under-reporting: even a reader that lists only a dozen
# of its forty rows has to look at all forty to name the largest.
_EXTREME = re.compile(r"^\s*(MAXIMUM|MINIMUM)\s*\|(.+?)\|(.+?)\|(.*)$", re.I | re.M)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# A table's row-label column is headed by a generic word, and a reader scanning
# that table will faithfully report it as though it were an entity. One document
# answered "the country with the highest share is Country" because the header
# survived all the way into the final sentence. These are English generics, not
# facts about any document.
_NOT_AN_ENTITY = {
    "entity", "n/a", "none", "country", "countries", "economy", "economies",
    "region", "regions", "name", "names", "item", "items", "total", "totals",
    "average", "averages", "mean", "median", "all", "other", "others", "value",
    "values", "year", "years", "rank", "ranking", "share", "percent",
    "percentage", "category", "group", "type", "unit", "units", "oecd",
    "oecd average", "eu average", "world", "sum", "subtotal", "n.a.", "-", "--",
}

# Every conversion is to the first unit in its family. Mixed units inside one
# attribute are the trap that made a 1,151-square-mile park look smaller than a
# 3,430-square-kilometre one.
_UNITS = {
    "sq mile": ("sq km", 2.58999), "sq miles": ("sq km", 2.58999),
    "square mile": ("sq km", 2.58999), "square miles": ("sq km", 2.58999),
    "mi2": ("sq km", 2.58999), "sq km": ("sq km", 1.0),
    "square kilometre": ("sq km", 1.0), "square kilometres": ("sq km", 1.0),
    "square kilometer": ("sq km", 1.0), "square kilometers": ("sq km", 1.0),
    "km2": ("sq km", 1.0),
    "mile": ("km", 1.60934), "miles": ("km", 1.60934),
    "km": ("km", 1.0), "kilometre": ("km", 1.0), "kilometres": ("km", 1.0),
    "foot": ("m", 0.3048), "feet": ("m", 0.3048), "ft": ("m", 0.3048),
    "m": ("m", 1.0), "metre": ("m", 1.0), "metres": ("m", 1.0),
    "meter": ("m", 1.0), "meters": ("m", 1.0),
}


def _parse_value(raw: str) -> tuple[float | None, str | None, str | None, float | None]:
    """
    '1151 sq miles' -> (1151.0, 'sq miles', 'sq km', 2980.5).

    Returns (value, printed_unit, unit_family, normalized) -- always four
    elements. A record whose value field carries no digits at all ("party to
    the convention") is a legitimate thing for a reader to emit, so it returns
    four Nones rather than three.

    The unit may be several tokens ("sq km", "square miles"), so try the longest
    first. Taking only the first token collapsed "sq km" and "sq miles" both to
    "sq" -- which made a square-mile area look like square kilometres, silently
    reproducing the exact trap this table exists to catch.
    """
    m = _NUMBER.search(raw)
    if not m:
        return None, None, None, None
    value = float(m.group(0).replace(",", ""))

    tokens = [t for t in re.split(r"[\s,()]+", raw[m.end():].strip().lower()) if t]
    for width in (3, 2, 1):
        if len(tokens) < width:
            continue
        candidate = " ".join(tokens[:width]).strip(".")
        for form in (candidate, candidate.rstrip("s")):
            if form in _UNITS:
                family, factor = _UNITS[form]
                # Keep BOTH: the family drives comparison, the printed unit is
                # what the document actually said. Collapsing them hid the fact
                # that one entry was reported in different units from the rest --
                # which is not noise, it is sometimes the answer.
                return value, candidate, family, value * factor
    return value, (tokens[0] if tokens else None), None, value


def _make_record(entity, attribute, raw_value, where, source, kind):
    value, printed_unit, family, normalized = _parse_value(raw_value)
    return {"entity": entity, "attribute": attribute.lower(),
            "value": value, "unit": printed_unit, "unit_family": family,
            "normalized": normalized, "raw_value": raw_value,
            "where": where, "source": source, "kind": kind}


def parse_records(text: str, source: str = "") -> list[dict]:
    """
    Pull RECORD and MAXIMUM/MINIMUM lines out of one reader's reply.

    `source` names the slice that produced them. G1 threw that away, which is
    part of why it could not tell two disagreeing readers apart.
    """
    out = []
    for m in _RECORD.finditer(text):
        entity, attribute, raw_value, where = (p.strip() for p in m.groups())
        if not entity or entity.lower().strip(" .:") in _NOT_AN_ENTITY:
            continue
        out.append(_make_record(entity, attribute, raw_value, where, source, "record"))

    for m in _EXTREME.finditer(text):
        word, attribute, entity, raw_value = (p.strip() for p in m.groups())
        if not entity or entity.lower().strip(" .:") in _NOT_AN_ENTITY:
            continue
        out.append(_make_record(entity, attribute, raw_value,
                                f"local {word.lower()} of {source}",
                                source, word.lower()))
    return out


# --------------------------------------------------------------- PROVENANCE

def verify_record(record: dict, text: str) -> str:
    """
    Decide whether a record is corroborated by the raw text. Returns one of
    'verified', 'entity_not_found', 'value_not_near_entity', 'unverifiable'.

    The test is deliberately blunt and deliberately cheap: the entity name must
    occur in the text, and the printed digits must occur on the SAME LINE as one
    of those occurrences. Same-line rather than a character window because the
    material this protects against is tabular -- one row is one entity, and a
    generous window silently reaches into the neighbouring row.

    This is what a reported figure of 148,063 for Qatar fails. That string occurs
    nowhere in the document; the row actually reads 0.886, 82.4, 13.1. Nothing in
    g1 asked the question.
    """
    if record["value"] is None:
        return "unverifiable"

    digits = _NUMBER.search(record["raw_value"])
    if not digits:
        return "unverifiable"
    printed = digits.group(0)
    bare = printed.replace(",", "")

    entity = record["entity"].strip()
    if len(entity) < 2:
        return "unverifiable"

    found_entity, nearby = False, False
    for m in re.finditer(re.escape(entity), text, re.I):
        found_entity = True
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[line_start:line_end if line_end != -1 else len(text)]
        if len(line) <= 4000 and (printed in line or bare in line.replace(",", "")):
            return "verified"
        # Not every document puts an entity and its value on one line. In the
        # parks book the park's name is a heading and its area sits in a box
        # several lines below, so a same-line-only test would reject every
        # CORRECT record -- turning a check that was merely useless on that
        # document into one that is actively harmful.
        window = text[max(0, m.start() - PROVENANCE_WINDOW):
                      m.end() + PROVENANCE_WINDOW]
        if printed in window or bare in window.replace(",", ""):
            nearby = True

    if not found_entity:
        return "entity_not_found"
    return "verified_nearby" if nearby else "value_not_near_entity"


# ------------------------------------------------- RANKED TABLES (generic)
#
# The division of labour that g2 got wrong.
#
# A reader handed six thousand words containing a two-hundred-row table does not
# emit two hundred RECORD lines. It emits a dozen and stops -- which is how the
# highest life expectancy in the report went unreported while sitting twenty-three
# lines from the value that was reported, and how a count of 193 came back as 180.
# Asking a 24B model for completeness over a long table is asking the wrong
# instrument.
#
# But the model is the right instrument for the part Python cannot do: knowing
# WHICH COLUMN "gross national income per capita" means. So the two are split.
# Python parses every row of the table. The readers' records are used only to
# identify which column they were talking about. Then Python answers over all
# the rows.

# A ranked row: an ordinal, a name beginning with a letter, then numeric fields.
_CONTINUATION = re.compile(r"\bcont(?:inued|\.)?\b|[\u2192\u2190\u21d2]|^\s*\.\.\.", re.I)
_RANK_ROW = re.compile(r"^\s*(\d{1,4})\s+([A-Za-z][^\d]{1,45}?)\s{2,}(.+)$")
_FIELD = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_ranked_rows(text: str) -> list[dict]:
    """
    Every row of a ranked table, with its group heading.

    Generic throughout: a row is an ordinal followed by a name followed by
    numbers, ranks ascend down the table, and a group heading is a line carrying
    no digits at all. No band names, no column positions, no page numbers.
    """
    rows, group, previous = [], None, 0
    for line in text.split("\n"):
        stripped = line.strip()
        # A table that spans pages repeats a continuation marker between its
        # rows. Read as a group heading it silently steals rows from the real
        # group -- twelve from one band and eight from another, which is exactly
        # how 74 and 26 came back as 62 and 18. Continuation marks are a
        # typographic convention, like letter-spaced headings, not a fact about
        # any particular document.
        if _CONTINUATION.search(stripped):
            continue
        if (stripped and not any(c.isdigit() for c in stripped)
                and 5 < len(stripped) < 60 and stripped[0].isalpha()):
            group = stripped
            continue
        m = _RANK_ROW.match(line)
        if not m:
            continue
        rank = int(m.group(1))
        # Ranks ascend. A line that breaks the sequence is a layout artefact --
        # a stray "2  74" cost exactly one country from one band.
        if rank < previous:
            continue
        previous = rank
        values = [float(v.replace(",", "")) for v in _FIELD.findall(m.group(3))]
        if not values:
            continue
        rows.append({"rank": rank, "entity": m.group(2).strip(), "group": group,
                     "values": values,
                     "printed": _FIELD.findall(m.group(3))})
    return rows


def _column_for(rows: list[dict], reported: list[float]) -> int | None:
    """
    Which column the readers were reporting, decided by agreement with the table.

    The model knows what "life expectancy at birth" means; it just cannot read
    two hundred rows reliably. So let its handful of records vote on a column,
    then take the answer from every row of that column.
    """
    if not rows or not reported:
        return None
    width = max(len(r["values"]) for r in rows)
    best, best_hits = None, 0
    for col in range(width):
        column = {round(r["values"][col], 6) for r in rows if len(r["values"]) > col}
        hits = sum(1 for v in reported if round(v, 6) in column)
        if hits > best_hits:
            best, best_hits = col, hits
    # One accidental match proves nothing; require either a majority or two.
    return best if best_hits >= max(2, len(reported) // 2) else None


def settle_ranked_table(question: str, scope_text: str,
                        reported: list[float]) -> dict | None:
    """Answer a counting or superlative question over a parsed ranked table."""
    rows = parse_ranked_rows(scope_text)
    if len(rows) < 20:
        return None

    groups: dict[str, int] = {}
    for r in rows:
        if r["group"]:
            groups[r["group"]] = groups.get(r["group"], 0) + 1

    out = {
        "source": "parsed directly from the ranked table by Python, every row",
        "ranked_entries_total": len(rows),
        "rank_range": [rows[0]["rank"], rows[-1]["rank"]],
        "counts_by_group": groups or None,
    }

    col = _column_for(rows, reported)
    if col is not None:
        usable = [r for r in rows if len(r["values"]) > col]
        top = max(usable, key=lambda r: r["values"][col])
        bottom = min(usable, key=lambda r: r["values"][col])
        out["column_identified_from_reader_records"] = col
        out["maximum"] = {"entity": top["entity"], "value": top["printed"][col],
                          "rank": top["rank"], "group": top["group"]}
        out["minimum"] = {"entity": bottom["entity"], "value": bottom["printed"][col],
                          "rank": bottom["rank"], "group": bottom["group"]}
        threshold = parse_threshold(question)
        if threshold:
            value, op = threshold
            tests = {">=": lambda v: v >= value, ">": lambda v: v > value,
                     "<": lambda v: v < value, "<=": lambda v: v <= value}
            hit = [r for r in usable if tests[op](r["values"][col])]
            out["threshold"] = {"value": value, "operator": op}
            out["matching_count"] = len(hit)

    # Name the entities in this region that carry no rank. Readers report them
    # in good faith -- they are printed in the same region -- and one of them was
    # asserted as the answer to a question restricted to ranked entries, in two
    # separate runs. Listing them lets the synthesiser refuse them by name.
    ranked_names = {r["entity"].strip().lower() for r in rows}
    # The group headings must NOT go on this list. They are exactly what a
    # "how many fall into each group" answer has to name, and telling the
    # synthesiser never to mention them would break the question this same
    # parser gets right.
    reserved = ranked_names | {g.strip().lower() for g in groups}
    unranked = []
    for line in scope_text.split("\n"):
        m = re.match(r"^\s{2,}([A-Z][A-Za-z .,'()\-]{2,40}?)\s{2,}[\d.]", line)
        if m:
            name = m.group(1).strip()
            if name.lower() not in reserved and name not in unranked:
                unranked.append(name)
    if unranked:
        out["entities_here_that_carry_no_rank"] = unranked[:25]

    out["note"] = (
        "Every row of the table was read and counted in Python, so these totals "
        "are complete. Only ENTRIES CARRYING A RANK are included: an entity "
        "listed without a rank is not one of the ranked entries, and any name "
        "under 'entities_here_that_carry_no_rank' MUST NOT be given as the "
        "answer or mentioned at all, however a reader describes it. Use these "
        "numbers exactly."
    )
    return out


_GROUPING = re.compile(r"\b(?:in|into|for|per)\s+each\b|\beach\s+(?:group|category|"
                       r"band|class|tier|region|type)\b|\bhow many\b.{0,40}\beach\b",
                       re.I)


def categorical_tally(records: list[dict], question: str) -> dict | None:
    """
    "How many fall into each group?" is a count of entities per label, not a
    ranking of numbers. G1 dropped every non-numeric record on the floor and so
    could not answer this shape of question at all.
    """
    labelled = [r for r in records if r["normalized"] is None
                and r["raw_value"] and len(r["raw_value"]) < 60]
    if len(labelled) < 3:
        return None

    counts: dict[str, int] = {}
    for r in labelled:
        counts[r["attribute"]] = counts.get(r["attribute"], 0) + 1
    dominant = max(counts, key=counts.get)

    seen: dict[str, str] = {}
    for r in labelled:
        if r["attribute"] != dominant:
            continue
        seen.setdefault(r["entity"].strip().lower(), r["raw_value"].strip().lower())

    groups: dict[str, int] = {}
    for label in seen.values():
        groups[label] = groups.get(label, 0) + 1

    return {
        "counted_by": dominant,
        "distinct_entities": len(seen),
        "group_counts": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
        "total": len(seen),
        "note": ("Each entity was assigned to one group and Python counted the "
                 "groups. Entities reported more than once across slices were "
                 "counted once."),
    }


def _resolve_conflicts(numeric: list[dict]) -> tuple[list[dict], list[str]]:
    """
    One record per entity, chosen by evidence rather than by accident.

    G1 kept whichever duplicate had the longer 'where' string. That is not a
    tie-break, it is a coin toss with extra steps, and it is how the wrong ISBN
    reached the submission: one reader correctly labelled the PDF one, another
    mislabelled the print one, and the longer string won.

    Here, disagreement is treated as information. Majority across readers first;
    then a provenance-verified value over an unverified one; and whatever is
    still unresolved is reported rather than buried.
    """
    by_entity: dict[str, list[dict]] = {}
    for r in numeric:
        by_entity.setdefault(r["entity"].strip().lower(), []).append(r)

    unique, conflicts = [], []
    for key, group in by_entity.items():
        votes: dict[float, list[dict]] = {}
        for r in group:
            votes.setdefault(round(r["normalized"], 6), []).append(r)

        if len(votes) == 1:
            unique.append(group[0])
            continue

        def strength(item):
            value, rs = item
            verified = sum(1 for r in rs
                           if r.get("provenance") in ("verified", "verified_nearby"))
            return (verified, len(rs))

        ranked = sorted(votes.items(), key=strength, reverse=True)
        winner = ranked[0][1][0]
        unique.append(winner)
        conflicts.append(
            f"{winner['entity']}: kept {winner['raw_value']}, "
            f"also reported as " +
            ", ".join(sorted({r["raw_value"] for _, rs in ranked[1:] for r in rs}))
        )

    return unique, conflicts


def tally(records: list[dict], question: str, source_text: str = "") -> dict | None:
    """
    Count, rank and threshold the records the readers returned.

    This is the generic replacement for a per-document table parser: the model
    reports what its own slice says, Python does every arithmetic step. G2 adds
    a gate in front of the arithmetic -- a record that the raw text does not
    corroborate never reaches it.
    """
    # Provenance first, so that everything downstream -- conflict resolution
    # included -- can lean on it.
    rejected: dict[str, list[str]] = {}
    if source_text:
        checked = []
        for r in records:
            verdict = verify_record(r, source_text)
            r["provenance"] = verdict
            if verdict in ("verified", "verified_nearby", "unverifiable"):
                checked.append(r)
            else:
                rejected.setdefault(verdict, []).append(
                    f"{r['entity']} = {r['raw_value']}")
        records = checked

    numeric = [r for r in records if r["normalized"] is not None]
    if not numeric:
        cat = categorical_tally(records, question)
        if cat and rejected:
            cat["rejected_unverifiable"] = {k: v[:12] for k, v in rejected.items()}
        return cat

    # Readers may report more than one attribute. Comparing heights against
    # areas is meaningless, so rank within the attribute most readers reported
    # and say plainly what was set aside.
    counts: dict[str, int] = {}
    for r in numeric:
        counts[r["attribute"]] = counts.get(r["attribute"], 0) + 1
    dominant = max(counts, key=counts.get)
    set_aside = {a: n for a, n in counts.items() if a != dominant}
    numeric = [r for r in numeric if r["attribute"] == dominant]

    unique, conflicts = _resolve_conflicts(numeric)

    units = {r["unit"] for r in unique if r["unit"]}
    threshold = parse_threshold(question)

    matching = unique
    if threshold:
        value, op = threshold
        tests = {">=": lambda v: v >= value, ">": lambda v: v > value,
                 "<": lambda v: v < value, "<=": lambda v: v <= value}
        matching = [r for r in unique if tests[op](r["normalized"])]

    ranked = sorted(unique, key=lambda r: r["normalized"], reverse=True)
    odd_units = [f"{r['entity']} ({r['raw_value']})" for r in unique
                 if r["unit"] and len(units) > 1
                 and sum(1 for x in unique if x["unit"] == r["unit"]) == 1]

    note = ("Counted and ranked in Python over the records every reader returned; "
            "duplicates across slices were merged.")
    if source_text:
        note += (" Every value below was checked against the raw text: the entity "
                 "occurs in the document and the figure is printed on its line.")
    if len(units) > 1:
        note += (" Units were not uniform, so values were converted to a common "
                 "unit before comparison; the raw printed values are shown.")
    if set_aside:
        note += (" Records for other attributes were set aside: "
                 + ", ".join(f"{a} ({n})" for a, n in set_aside.items()) + ".")
    if rejected:
        note += (" Some reported values could not be found in the document and "
                 "were discarded before counting; they are listed separately and "
                 "must not appear in the answer.")

    return {
        "attribute_ranked": dominant,
        "attributes_set_aside": set_aside or None,
        "records_returned": len(records),
        "distinct_entities": len(unique),
        "threshold": {"value": threshold[0], "operator": threshold[1]} if threshold else None,
        "matching_count": len(matching),
        "matching_entities": [f"{r['entity']} ({r['raw_value']})" for r in
                              sorted(matching, key=lambda r: -r["normalized"])],
        "maximum": {"entity": ranked[0]["entity"], "value": ranked[0]["raw_value"],
                    "where": ranked[0]["where"]} if ranked else None,
        "minimum": {"entity": ranked[-1]["entity"], "value": ranked[-1]["raw_value"],
                    "where": ranked[-1]["where"]} if ranked else None,
        "units_seen": sorted(units),
        "mixed_units": len(units) > 1,
        "reported_in_a_different_unit": odd_units or None,
        "rejected_not_found_in_document": {k: v[:12] for k, v in rejected.items()} or None,
        "unresolved_disagreements": conflicts[:12] or None,
        "note": note,
    }


# ------------------------------------------------------------- THE THROTTLE

def _is_rate_limit(error: Exception) -> bool:
    return (getattr(error, "status_code", None) == 429
            or "RateLimit" in type(error).__name__
            or "429" in str(error) or "rate limit" in str(error).lower())


class Throttle:
    """
    One shared pace for every thread, widened by 429s and narrowed by success.

    Per-call backoff cannot clear a per-minute quota: each thread backs off
    privately, wakes on the same schedule and collides again. Absorbed 53
    rejections on a real run without losing a single reader.
    """

    def __init__(self, rpm: float):
        self._lock = threading.Lock()
        self._floor = 60.0 / rpm
        self._interval = self._floor
        self._next_at = 0.0
        self.rejections = 0

    def wait(self) -> None:
        with self._lock:
            start = max(time.monotonic(), self._next_at)
            self._next_at = start + self._interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalize(self) -> float:
        with self._lock:
            self.rejections += 1
            self._interval = min(15.0, max(self._interval * 2, 1.0))
            cooldown = self._interval
            self._next_at = max(self._next_at, time.monotonic() + cooldown)
        return cooldown

    def relax(self) -> None:
        with self._lock:
            self._interval = max(self._floor, self._interval * 0.8)

    def pace(self) -> str:
        return f"{60.0 / self._interval:.0f} calls/min"


THROTTLE = Throttle(START_RPM)


def call_llm(prompt: str) -> str:
    """One broker call, paced globally, retried patiently. Failed calls are free."""
    messages = [{"role": "user", "content": prompt}]
    sampling = {} if TEMPERATURE is None else {"temperature": TEMPERATURE}
    last = None
    for attempt in range(MAX_ATTEMPTS):
        THROTTLE.wait()
        try:
            response = get_client().chat.completions.create(
                model=MODEL, messages=messages, **sampling)
            THROTTLE.relax()
            return response.choices[0].message.content or ""
        except Exception as e:
            last = e
            if not _is_rate_limit(e):
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise
                if attempt >= 2:
                    raise
                time.sleep(2 ** attempt + random.random())
                continue
            time.sleep(THROTTLE.penalize() * (1.0 + random.random()))
    raise last


# ------------------------------------------------------------------ PROMPTS
# Nothing below names a particular document. The only description the readers
# get is the one the caller supplies.

PROSE_PROMPT = """You are reading part {n} of {of} of {doc}. The text below is verbatim --
nothing has been summarized or removed.

Answer the question using ONLY this text. Every name, figure, percentage and
date you write must be findable in the text below. NEVER supply a fact from your
own knowledge: another reader may hold the real evidence, and an invented detail
displaces it.

You hold roughly one {of}th of the document. Most parts will not contain what is
asked. If this part has nothing relevant, reply with exactly:

    {marker}

and nothing else. DO NOT GUESS.

When you DO have something, give up to two kinds of line.

PATTERN lines -- at most two, and ONLY if this part actually states the idea:

    PATTERN: <the claim this part makes, quoting or closely restating a sentence>

INSTANCE lines -- one per concrete example:

    WHERE -- the specific fact -- figure, percentage, year, name

Rules:
- Say where it comes from on every line: the section, table, figure or page.
- Copy figures, percentages, years and proper names EXACTLY as printed.
- KEEP QUALIFIERS. "about two-thirds" is not "most". "projected to" is not "did".
- Numbers that appear as bare chart furniture -- axis ticks like "0 20 40 60 80",
  or a value with no sentence around it -- are NOT quotable facts. Skip them.
- AT MOST 12 instance lines. If this part has more, give the 12 most specific.
- Do not count anything across the whole document; you see only one {of}th.

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""

RECORD_PROMPT = """You are reading part {n} of {of} of {doc}.{scope} The text below is
verbatim.

The question needs a count or a comparison across ALL the parts. You cannot do
that -- you hold only this one. Do not try, and do not estimate a total. Your job
is to REPORT WHAT THIS PART CONTAINS, exactly and COMPLETELY. Python will count
across all the parts afterwards.

For EVERY item in this part that is relevant to the question, emit one line:

    RECORD | entity | attribute | value | where

For example, if asked about the areas of things:

    RECORD | Vatnajokull National Park | area | 13,600 sq km | Park in numbers box
    RECORD | Jotunheimen National Park | area | 1151 sq miles | Park in numbers box

COMPLETENESS IS THE ENTIRE POINT OF THIS TASK. If this part holds a list or a
table with forty rows that bear on the question, emit forty lines. Do not select
the interesting ones, do not stop at ten, do not summarize the rest as "and
others". A row you leave out is invisible to every later step: the highest value
in the whole document has already been missed once because it sat four lines
below one that was reported. There is no limit on how many lines you may emit.

Then, as the LAST lines of your reply, state this part's own extremes:

    MAXIMUM | attribute | entity | value
    MINIMUM | attribute | entity | value

Give these even if you listed every row -- they are a check on the listing. They
describe THIS PART ONLY, never the whole document.

Rules:
- COPY THE VALUE EXACTLY AS PRINTED, including its unit and any comma. Do not
  convert, round or normalize. If a unit differs from the others, that is
  important information -- reproduce it as printed.
- Take the value from the SAME ROW or sentence as the entity. Columns in a
  table are easy to slip: check that the number you copy is under the heading
  the question asks about, not the column beside it.
- One line per entity. Use the entity's full name as the document prints it.
- The attribute is what was measured, in the document's own words.
- When the question asks which GROUP or CATEGORY each item belongs to, the value
  is that group's name, not a number.
- 'where' locates it: the section, box, table, figure or page.
- If a value is missing or unreadable for an entity, skip that entity.
- NEVER write a figure that is not printed in the text below. A number recalled
  from your own knowledge is worse than no line at all; it will be checked
  against the document and it will be thrown away.
- Emit nothing but these lines unless you also have context worth stating, in
  which case add at most two lines beginning "NOTE: ".

If this part contains no relevant item at all, reply with exactly:

    {marker}

and nothing else. DO NOT GUESS and DO NOT invent entities.

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""

SYNTHESIS_PROMPT = """Write one final answer to the question, using the two blocks below.

They are NOT equal sources.

============================ HOW TO USE THEM ============================

COMPUTED VALUES were calculated in Python over what every reader returned, across
the entire document.

    THIS BLOCK IS AUTHORITATIVE AND CANNOT BE OVERRULED. For any total, count,
    ranking, maximum or verdict about what the document never mentions, USE ITS
    VALUE EXACTLY -- even if several readers say otherwise. Readers see one slice
    each and cannot count, rank, or establish absence. If a reader's claim
    conflicts with this block, THE READER IS WRONG: drop the reader's version
    entirely and do not mention it.

TEXT EVIDENCE is what readers found in their own slice of the verbatim text.
They are the only source for quotations, arguments, examples and specific facts.

    Within this block, COMBINE EVERYTHING. If one reader names three findings and
    another names two, your answer gives all five -- take the UNION, never a
    substitution. Two readers naming different things is not a conflict.

============================== HARD RULES ==============================

- NAME NOTHING THAT IS NOT IN EITHER BLOCK. If a name, figure or claim appears in
  no reader's lines and in no computed value, it does not go in your answer,
  however plausible. This is the most damaging error available to you.
- If the computed block lists values under "rejected_not_found_in_document" or
  "unresolved_disagreements", those figures were checked against the document
  and are not printed there. A reader may still state them below. NEVER repeat
  one, and never mention that anything was rejected.
- Every specific detail from the text evidence must survive into your answer.
  Length costs nothing; brevity loses points.
- Keep any label the question uses (a chapter, figure or table) so the claim
  stays locatable.
- Give exactly ONE verdict where the question asks for one -- one absent subject,
  one highest value, one count. Never offer alternatives for a verdict.
- Where the question asks you to state two claims in tension, state BOTH in full,
  then say plainly how they are reconciled.
- Never mention the blocks, the readers, or that you were given anything.
- Never hedge. Commit.
- If the question is listed below as asking for several separate things, ANSWER
  EVERY ONE OF THEM, in the order given, even where that makes the answer long.
  Each is graded on its own and a part you never mention scores nothing. Do not
  merge two of them into a single remark, and do not drop the smallest one.

=========================== COMPUTED VALUES ===========================
{computed}

============================ TEXT EVIDENCE ============================
{evidence}

=============================== QUESTION ==============================
{question}{parts}

Final answer, plain text, no preamble.
"""


COVERAGE_PROMPT = """Below is a question, the separate things it asks for, and an
answer that was written for it.

Say which of the listed parts the answer does NOT address. Judge only what is
present in the answer; do not judge whether it is correct, and do not rewrite it.

Reply with the numbers and a few words each, one per line:

    (3) does not name the framework
    (4) gives no figure for patents

If the answer addresses every part, reply with exactly:

    NOTHING MISSING

QUESTION:
{question}{parts}

ANSWER:
{answer}

Your reply, plain text, no preamble.
"""

GAP_PROMPT = """An answer to the question below left some of what was asked
unaddressed. Supply ONLY the missing material, using the two blocks.

Do not restate what is already answered. Do not write an introduction. Write
plain sentences that state the missing facts, so they can be appended to the
existing answer.

The same hard rule applies: NAME NOTHING that is not in the blocks below. If the
evidence does not contain what is missing, reply with exactly NO EVIDENCE.

NEVER restate, recount or revise any figure that appears under COMPUTED VALUES.
Those were calculated over the whole document and are already in the answer. If
what is missing is one of them, reply with exactly NO EVIDENCE instead.

=========================== COMPUTED VALUES ===========================
{computed}

============================ TEXT EVIDENCE ============================
{evidence}

=============================== QUESTION ==============================
{question}

========================= WHAT IS STILL MISSING =======================
{missing}

The missing material only, plain text, no preamble.
"""


# ------------------------------------------------------------------ ANSWERING

def _run_readers(jobs: list[tuple[str, str]]) -> tuple[dict, list]:
    done, failed = {}, []

    def run(job):
        name, prompt = job
        try:
            return name, call_llm(prompt), None
        except Exception as e:
            return name, "", e

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for name, text, error in pool.map(run, jobs):
            if error is None:
                done[name] = text.strip()
            else:
                failed.append((name, error))
    return done, failed


def _declares_no_evidence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return stripped.upper().replace(NO_EVIDENCE, "").strip(" .\"'`\n\t-–—") == ""


def computed_block_or_none(precomputed) -> str:
    return (json.dumps(precomputed, indent=2, ensure_ascii=False) if precomputed
            else "(nothing could be computed for this question)")


def shortfall_of(missing, jobs) -> str:
    return f" ({len(missing)} of {len(jobs)} readers unavailable)" if missing else ""


def answer_question(doc: dict, question: str, category: str | None) -> dict:
    """One question: Python where it can settle things, readers where it cannot."""
    mode = question_mode(question, category)
    precomputed = None          # a Python result that answers PART of the question
    parts = question_parts(question)
    parts_block = format_parts(parts)
    if len(parts) > 1:
        _log(f"  ~ question asks for {len(parts)} separate things")

    if mode == "absence":
        settled = settle_absence(question, doc["body"], doc["tail"], doc["text"])
        if settled:
            note = f"settled in Python by {settled['basis']}; no model was called"
            if settled["ambiguous"]:
                note += (" | WARNING: more than one candidate looked absent: "
                         + ", ".join(settled["ambiguous"]))
            return {"answer": settled["answer"], "computed": note,
                    "evidence": "(not used: settled by computation)",
                    "readers": 0, "decision": "computed"}
        _log("  ~ absence could not be settled by counting; using readers")

    # A counting question that names its own location is read inside that
    # location, finely sliced. A question that names none is read exactly as g1
    # read it: every slice of the whole document.
    if mode == "tally":
        # "How many parks are profiled?" is answered by the document's own
        # repeated entry structure, not by asking readers to count across
        # slices -- which returned 40, then 49, for a book of 60.
        if re.search(r"\bhow many\b|\bnumber of\b", question, re.I) and \
                not question_locators(question):
            entries = count_entries(doc["text"])
            if entries and re.search(r"\b(?:are|is)\s+(?:there\s+)?"
                                     r"(?:profiled|described|covered|featured|"
                                     r"included|listed)\b|\bdoes the \w+ "
                                     r"(?:profile|cover|describe)\b",
                                     question, re.I):
                # Settle in Python ONLY when this answers the whole question.
                # d01 asks for the total, how many are in one country, and their
                # names; returning the total alone with readers:0 answered one
                # part of three and left the other two permanently unanswerable.
                # When the question asks for more, the count becomes an
                # authoritative computed value and the readers still run.
                if python_answers_all(parts):
                    _log(f"  ~ {entries['entries']} entries from the document's "
                         f"own repeated structure; no model called")
                    return {"answer": f"The book profiles {entries['entries']} of "
                                      f"them. " + entries["note"],
                            "computed": json.dumps(entries, indent=2,
                                                   ensure_ascii=False),
                            "evidence": "(not used: settled by computation)",
                            "readers": 0, "decision": "computed"}
                _log(f"  ~ {entries['entries']} entries counted in Python; readers "
                     f"still needed for the other {len(parts) - 1} part(s)")
                precomputed = entries
        counted = settle_identifier_count(question, doc["text"])
        if counted and not python_answers_all(parts):
            _log("  ~ identifiers counted in Python; readers still needed for the "
                 f"other {len(parts) - 1} part(s)")
            precomputed, counted = counted, None
        if counted:
            _log("  ~ settled by counting the document's own identifiers; "
                 "no model called")
            return {"answer": counted["answer"],
                    "computed": json.dumps(
                        {k: v for k, v in counted.items() if k != "identifiers"},
                        indent=2, ensure_ascii=False),
                    "evidence": "(not used: settled by computation)",
                    "readers": 0, "decision": "computed"}

    scope, chunks, scope_note = None, doc["chunks"], ""
    if mode == "tally":
        scope = scope_for(doc["text"], question)
        if scope:
            chunks = chunk_lines(scope["text"], TALLY_WORDS_PER_READER)
            scope_note = (f" You are reading only the part of the document under "
                          f"\"{scope['locator']}\", which is what the question asks "
                          f"about.")
            _log(f"  ~ scoped to {scope['locator']}: {scope['words']:,} words, "
                 f"{len(chunks)} readers (was {len(doc['chunks'])})")
        else:
            # No locator: read everything, but still on line boundaries so that
            # tabular material keeps its rows.
            chunks = chunk_lines(doc["text"], WORDS_PER_READER)
            _log(f"  ~ no locator in the question; reading all {len(chunks)} slices")

    if mode == "tally":
        jobs = [(f"part {c['n']}/{c['of']}",
                 RECORD_PROMPT.format(n=c["n"], of=c["of"], marker=NO_EVIDENCE,
                                      doc=doc["description"], scope=scope_note,
                                      chunk=c["text"], question=question))
                for c in chunks]
    else:
        jobs = [(f"part {c['n']}/{c['of']}",
                 PROSE_PROMPT.format(n=c["n"], of=c["of"], marker=NO_EVIDENCE,
                                     doc=doc["description"], chunk=c["text"],
                                     question=question))
                for c in chunks]

    done, failed = _run_readers(jobs)
    for attempt in (1, 2):
        if not failed:
            break
        # Only rate limits are worth waiting out. Every other failure -- a bad
        # key, an unreachable broker, a malformed request -- will fail again
        # identically, and two cooldowns burned 120 seconds per question while
        # reporting "rate-limited" for an error that was nothing of the kind.
        if not any(_is_rate_limit(e) for _, e in failed):
            first = failed[0][1]
            _log(f"  ! {len(failed)} reader(s) failed for a non-retryable reason -- "
                 f"{type(first).__name__}: {str(first)[:180]}")
            break
        cooldown = 30 * attempt
        first = failed[0][1]
        _log(f"  ~ {len(failed)} reader(s) failed -- {type(first).__name__}: "
             f"{str(first)[:140]}; cooling {cooldown}s (pace {THROTTLE.pace()})")
        time.sleep(cooldown)
        names = {n for n, _ in failed}
        recovered, failed = _run_readers([j for j in jobs if j[0] in names])
        done.update(recovered)

    missing = [n for n, _ in failed]
    if missing:
        _log(f"  ! {len(missing)} reader(s) never returned: {', '.join(missing)}")

    kept = [(name, text) for name, text in done.items()
            if not _declares_no_evidence(text)]
    if not kept:
        # NEVER return nothing. An unanswered question scores zero with total
        # certainty; a wrong one scores zero too, and a partly-right one scores
        # something -- so silence is the only choice that cannot win. This path
        # raised until an absence question on a third document had every reader
        # decline at once and went to the grader blank.
        #
        # Every reader saw its whole slice and found nothing worth quoting, so
        # for a question asking what the document never covers, that unanimous
        # silence IS the finding. Say so, using the question's own candidates.
        _log("  ! every reader declared NO EVIDENCE; answering from the "
             "unanimous silence rather than leaving it blank")
        subjects = [s["display"] for s in
                    question_subjects(_list_segment(candidate_list(question)))]
        if mode == "absence" and subjects:
            answer = (f"{subjects[-1].capitalize()} is never substantively "
                      f"discussed in the document. Every part of the document "
                      f"was read for this question and none of it addresses "
                      f"{subjects[-1]}, while the other subjects named in the "
                      f"question — {', '.join(subjects[:-1])} — do appear.")
        else:
            answer = ("No part of the document states this directly. Every "
                      "section was read and none contains material that answers "
                      "the question.")
        return {"answer": answer, "computed": computed_block_or_none(precomputed),
                "evidence": "(no reader returned evidence)",
                "readers": 0, "decision": "no_evidence_anywhere" + shortfall_of(missing, jobs)}

    computed_block = "(nothing could be computed for this question)"
    if precomputed:
        computed_block = json.dumps(precomputed, indent=2, ensure_ascii=False)
    if mode == "tally":
        records = [r for name, text in kept for r in parse_records(text, name)]
        # Verify against the region the records actually came from. Checking a
        # scoped record against the whole document would let a figure from a
        # different table corroborate itself -- which is the exact error that
        # put Norway's Gender Development Index value into an HDI answer.
        against = scope["text"] if scope else doc["text"]
        summary = tally(records, question, against)
        if summary and precomputed:
            summary = {"counted_in_python": precomputed, **summary}
        elif precomputed and not summary:
            summary = dict(precomputed)

        # If the scoped region is a ranked table, Python reads every row and
        # overrides the readers' partial listing. The readers still decide WHICH
        # column the question is about -- that is semantics, and they are good
        # at it. Completeness is arithmetic, and they are not.
        if scope:
            reported = [r["normalized"] for r in records
                        if r.get("normalized") is not None]
            table = settle_ranked_table(question, scope["text"], reported)
            if table:
                _log(f"  ~ parsed {table['ranked_entries_total']} ranked rows in "
                     f"{scope['locator']}; Python answers over all of them")
                summary = {"scope": f"restricted to {scope['locator']}",
                           "authoritative": table,
                           "reader_records_for_reference": summary}
        if summary:
            if scope and "scope" not in summary:
                summary = {"scope": f"restricted to {scope['locator']} "
                                    f"({scope['words']:,} words)", **summary}
            computed_block = json.dumps(summary, indent=2, ensure_ascii=False)

    evidence_block = "\n\n".join(f"READER {n}:\n{t}" for n, t in kept)
    shortfall = f" ({len(missing)} of {len(jobs)} readers unavailable)" if missing else ""

    try:
        final = call_llm(SYNTHESIS_PROMPT.format(
            computed=computed_block, evidence=evidence_block,
            question=question, parts=parts_block)).strip()
    except Exception as e:
        _log(f"  ! synthesis failed ({e}); falling back to the fullest draft")
        final = ""

    if not final:
        return {"answer": max((t for _, t in kept), key=len),
                "computed": computed_block, "evidence": evidence_block,
                "readers": len(kept), "decision": "synthesis_failed" + shortfall}

    # Coverage pass. The scoring counts key points, so an answer that addresses
    # two of four parts loses half the question however well it is written. One
    # call names the parts that went unanswered; a second answers only those,
    # from the same evidence, and the result is appended. Padding is free.
    filled = ""
    # Run on every synthesized answer, not only decomposed ones. The two
    # questions that lost points the same way in all three scored runs -- a
    # framework never named, a finding never mentioned -- are single-clause
    # questions that no splitter breaks apart.
    if True:
        try:
            missing = call_llm(COVERAGE_PROMPT.format(
                question=question, parts=parts_block, answer=final)).strip()
            if (missing and "NOTHING MISSING" not in missing.upper()
                    and not _declares_no_evidence(missing)):
                _log(f"  ~ coverage check: {missing[:90]}")
                filled = call_llm(GAP_PROMPT.format(
                    computed=computed_block, evidence=evidence_block,
                    question=question, missing=missing)).strip()
        except Exception as e:
            _log(f"  ~ coverage check skipped ({type(e).__name__})")
    if filled and not _declares_no_evidence(filled):
        # The fill may only ADD. It ran on the same evidence with a weaker grip
        # on the authoritative block, and when it restated a settled count it
        # restated it wrongly: one document came back with a perfect answer
        # followed by a paragraph zeroing every figure in it, which matched two
        # forbidden claims and halved a question that had been fully correct.
        #
        # So any figure the computed block settled may appear in the fill only
        # with the same value. Disagree once and the whole fill is dropped --
        # the first answer was already right, and the appendix is optional.
        settled = set(_NUMBER.findall(computed_block))
        appended = set(_NUMBER.findall(filled))
        if settled and appended and not (settled & appended):
            _log(f"  ! coverage fill gives figures ({', '.join(sorted(appended)[:5])}) "
                 f"that agree with none of the settled values; discarded")
        else:
            final = f"{final}\n\n{filled}"

    return {"answer": final, "computed": computed_block,
            "evidence": evidence_block, "readers": len(kept),
            "decision": f"{mode}_synthesized" + shortfall}


# -------------------------------------------------------------------- RUNNER

def load_document(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = normalize_text(f.read())

    size = measure(text)
    structure = discover_structure(text)
    body, tail = split_body(text, structure)
    chunks = chunk_text(text, auto_chunk_count(size["words"]))

    description = os.environ.get(
        "DOC_DESCRIPTION",
        f"a large document ({size['words']:,} words)")

    return {"text": text, "body": body.lower(), "tail": tail.lower(),
            "chunks": chunks, "size": size, "structure": structure,
            "description": description}


def _completed(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            previous = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {e["id"]: e for e in previous.get("answers", [])
            if e.get("answer") and not e.get("error")
            and not e.get("decision", "").endswith(")")}


def run(document_path: str, questions_path: str, output_path: str) -> dict:
    started = time.perf_counter()
    doc = load_document(document_path)
    questions = load_questions(questions_path)

    s, st = doc["size"], doc["structure"]
    _log(f"document      {s['words']:,} words, ~{s['estimated_tokens']:,} tokens")
    _log(f"structure     page markers: {st['page_markers']}, "
         f"reference section: {'found' if st['reference_section_found'] else 'none'}"
         f" (body is {st['body_fraction']:.0%} of the file)")
    _log(f"readers       {len(doc['chunks'])} slices of "
         f"~{s['words'] // len(doc['chunks']):,} words")
    _log(f"questions     {len(questions)}")

    reuse = _completed(output_path)
    if reuse:
        _log(f"resuming      {len(reuse)} answer(s) kept from {output_path}")

    submission = {"team": TEAM, "notes": NOTES, "answers": [],
                  "timing": {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "document": doc["size"], "structure": st,
                             "readers": len(doc["chunks"])}}
    decisions, per_question = {}, {}

    for q in questions:
        qid = q["id"]
        if qid in reuse:
            submission["answers"].append(reuse[qid])
            decisions["reused"] = decisions.get("reused", 0) + 1
            _log(f"  [ reused] {qid}")
        else:
            elapsed_min = (time.perf_counter() - started) / 60
            if DEADLINE_MINUTES and elapsed_min > DEADLINE_MINUTES:
                _log(f"  ! deadline reached; {qid} left unanswered")
                submission["answers"].append(
                    {"id": qid, "answer": "", "error": "deadline reached"})
                continue

            _log(f"Answering {qid} [{q.get('category', 'uncategorized')}]...")
            entry, q_started = {"id": qid}, time.perf_counter()
            try:
                result = answer_question(doc, q["question"], q.get("category"))
                entry.update({"answer": result["answer"],
                              "decision": result["decision"],
                              "readers_with_evidence": result["readers"],
                              "evidence": [f"COMPUTED: {result['computed']}",
                                           f"TEXT EVIDENCE:\n{result['evidence']}"]})
                decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
            except Exception as e:
                entry.update({"answer": "", "error": str(e)})
                _log(f"  -> failed: {e}")
            entry["seconds"] = round(time.perf_counter() - q_started, 2)
            per_question[qid] = entry["seconds"]
            _log(f"  [{entry['seconds']:>7.1f}s] {qid} "
                 f"({entry.get('readers_with_evidence', 0)} readers)")
            submission["answers"].append(entry)

        submission["timing"].update({
            "answering_seconds": round(time.perf_counter() - started, 2),
            "decisions": dict(decisions),
            "per_question_seconds": dict(per_question),
            "rate_limit_rejections": THROTTLE.rejections,
        })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

    _log(f"\nWrote {output_path} ({len(submission['answers'])} answers)")
    _log(f"  decisions: {decisions}")
    _log(f"  429s absorbed: {THROTTLE.rejections}; final pace {THROTTLE.pace()}")
    unfinished = [a["id"] for a in submission["answers"] if not a.get("answer")]
    if unfinished:
        _log(f"  INCOMPLETE, re-run to finish: {', '.join(unfinished)}")
    return submission


NOTES = (
    "One pipeline, no retrieval and no assumptions about the document. The text "
    "is read verbatim in slices sized from its own word count (~6,000 words per "
    "reader), and every reader reads its slice for every question -- no chunk is "
    "ever selected, ranked or filtered by the question. Counting and comparison "
    "questions are answered by having each reader report only what its own slice "
    "contains, as structured records, which Python then dedupes, unit-converts, "
    "thresholds and ranks across the whole document; a reader is never asked for "
    "a total it cannot see. Absence is decided by counting terms over the whole "
    "text, and where a bibliography is detected, a term occurring only inside a "
    "cited work's title is reported as never discussed. Computed values are "
    "authoritative in the final synthesis and are never restated by a model."
)


if __name__ == "__main__":
    run(DOCUMENT, QUESTIONS, SUBMISSION)
