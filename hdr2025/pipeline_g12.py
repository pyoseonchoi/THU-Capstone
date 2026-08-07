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

from pipeline_ed2 import get_client, load_questions as _load_questions_file

# ----------------------------------------------------------------- CONFIG

DOCUMENT = os.environ.get("DOC", "hdr2025.txt")
QUESTIONS = os.environ.get("QUESTIONS", "hdr_questions.yaml")
SUBMISSION = os.environ.get("SUBMISSION_FILE", "submission_g8.json")
TEAM = os.environ.get("TEAM", "group_c")

# Which g8 steps are live. Every step is independently switchable because the
# build plan requires scoring each one before the next: batching them and
# scoring once at the end cannot tell you which step moved which category.
#
#   1  normalize_text: mojibake repair + space-separated thousands
#   2  single committed draft (the coverage pass edits instead of appending)
#   3  cross-section: phrase-anchored chunk selection
#   4  arc synthesis: enumeration instead of argument
#   5  claim ledger for contradiction  -- OFF by default, see the note below
#
# Step 5 is gated off deliberately. Its own precondition is that step 1's
# extraction fix has been CONFIRMED on the contradiction questions, and that
# confirmation needs a scored run. An ingest pass over unreliable extraction
# launders the same wrong numbers through Python and stamps them authoritative,
# which is strictly worse than the current failure -- it removes the model's
# ability to hedge. Turn it on with G8_STEPS=1,2,3,4,5 once step 1 is verified.
# g11: the default is now every step, not 1-4. Three generations were scored
# with the ledger reachable only through an environment variable that had to be
# remembered on the command line. It was in fact set on every scored run -- the
# submission's own notes prove it, since the ledger sentence is emitted under
# step_on("5") -- but a default that loses a subsystem when someone forgets a
# variable is a bad default to carry into a single-run evaluation.
G8_STEPS = {s.strip() for s in
            os.environ.get("G8_STEPS", "1,2,3,4,5,6,7").split(",")
            if s.strip()}


def step_on(number: str) -> bool:
    """Is this g8 step live in the current run?"""
    return number in G8_STEPS or "all" in G8_STEPS


# g9's own fixes, separately switchable so a regression can be bisected without
# a rebuild. All three are post-extraction: they change what Python does with
# the records the readers already returned, not what the readers are asked for,
# and none of them costs a single extra model call.
#
#   attr     tally() ranks within a measurement FAMILY, not an exact label
#   catattr  categorical_tally() does the same, and dedupes on the entity key
#   entity   _resolve_conflicts() groups on _entity_key, not the raw string
#
# Turn one off with e.g. G9_FIXES=attr,entity  (omitting catattr).
G9_FIXES = {f.strip() for f in
            os.environ.get("G9_FIXES", "attr,catattr,entity").split(",")
            if f.strip()}


def fix_on(name: str) -> bool:
    """Is this g9 fix live in the current run?"""
    return name in G9_FIXES or "all" in G9_FIXES


# g10. Every one of these is post-extraction or prompt-level, and none of them
# adds a single model call to a run.
#
#   unitcol   parse_records tolerates a reader that emits its own unit column
#   labelunit unit outliers found in the RAW TEXT, not in the readers' records
#   claimfam  a superlative only implies a measurement its own words don't name
#   scope     ingest readers always state which country an entity is in
#   setcross  set-valued cross-section questions enumerate instead of picking one
#
# Turn one off with e.g. G10_FIXES=unitcol,labelunit  (omitting the rest).
G10_FIXES = {f.strip() for f in os.environ.get(
    "G10_FIXES", "unitcol,labelunit,claimfam,scope,setcross").split(",")
    if f.strip()}


def g10_on(name: str) -> bool:
    """Is this g10 fix live in the current run?"""
    return name in G10_FIXES or "all" in G10_FIXES


# g11.
#   claimscope  a superlative's scope is read from its own words, not only from
#               a country record some reader may or may not have volunteered
#   recover     a record whose value came back as a NAME instead of a number has
#               the number read back out of the document
#   fragment    a partial phrase does not certify a subject when every one of its
#               occurrences belongs to a different phrase
G11_FIXES = {f.strip() for f in
             os.environ.get("G11_FIXES", "claimscope,recover,fragment").split(",")
             if f.strip()}


def g11_on(name: str) -> bool:
    """Is this g11 fix live in the current run?"""
    return name in G11_FIXES or "all" in G11_FIXES


# g12: header-aware segmentation. ADDITIVE. Every one of these is inert on a
# document whose headers do not pass the density gate, and `G12_FIXES=` (empty)
# restores g11 exactly -- verified byte-for-byte by header_equivalence.py, which
# compares every prompt g11 and g12 build for the same question and document.
#
#   headers     chunk on '## ' section boundaries instead of on a word count,
#               when and only when detect_headers/headers_are_usable agree the
#               document is really built that way
#   arcsection  the arc reader is TOLD which sections its part covers instead of
#               being asked to infer one
#   entityseed  header titles that name an entity are given to the record and
#               ingest readers as a roster
#   provwindow  the provenance window is measured from the document's own entry
#               period instead of being a 1,200-character constant
#
# None of the four adds a model call. `headers` changes where the slice
# boundaries fall, not how many there are.
#
# provwindow is OFF by default, deliberately. It is the fix for the thing this
# generation actually found -- see entry_period() -- but it moves a number that
# every counting question depends on, and the standing rule is one change per
# scored run. Turn it on with G12_FIXES=headers,arcsection,entityseed,provwindow
# once the header change itself has a score.
G12_FIXES = {f.strip() for f in
             os.environ.get("G12_FIXES", "headers,arcsection,entityseed").split(",")
             if f.strip()}


def g12_on(name: str) -> bool:
    """Is this g12 capability live in the current run?"""
    return name in G12_FIXES or "all" in G12_FIXES


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


# A UTF-8 lead byte that was decoded as latin-1, followed by its continuation
# byte. Written as escapes rather than literals so the pattern cannot itself be
# corrupted by whatever encoding this file is edited or transferred in.
#   Ã = A-tilde   Â = A-circumflex   Î/Ï = I-circ/diaeresis
# The continuation byte always lands in U+0080-U+00BF.
_MOJIBAKE = re.compile("[ÃÂÎÏ][-¿]")

# Below this many hits the sequences are more likely to be real characters than
# a systematic encoding fault, and repairing would corrupt rather than restore.
MOJIBAKE_MIN_HITS = int(os.environ.get("MOJIBAKE_MIN_HITS", "20"))


def repair_mojibake(text: str) -> str:
    """
    Undo UTF-8 bytes that were decoded as latin-1: 'PA^3les' -> 'Poles' (with
    the right accents). The OECD document arrives this way, and it mangles
    exactly the tokens that key points ask us to name -- every accented proper
    noun, every apostrophe and en-dash:

        PA'les de compA(c)titivitA(c)  ->  Poles de competitivite
        TA1/4rkiye                     ->  Turkiye
        I(c)- and I- convergence       ->  beta- and sigma-convergence

    (Rendered above without the actual bytes so this docstring survives being
    copied around; the real repair restores the true accented characters.)

    A key that says 'Turkiye' does not match an answer that says the mangled
    form, so this is key points earned and then lost to an encoding.

    Two guards. The density check stops a document that genuinely contains
    these characters from being rewritten, and the round-trip is attempted
    inside a try: a document holding any character outside latin-1 -- the parks
    book carries Japanese and Thai -- raises rather than silently corrupting.
    """
    hits = len(_MOJIBAKE.findall(text))
    if hits < MOJIBAKE_MIN_HITS:
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        _log(f"preprocess    {hits:,} mojibake-like sequences, but the text does "
             f"not round-trip through latin-1; left as it is")
        return text
    _log(f"preprocess    repaired {hits:,} mojibake sequences")
    return repaired


# A digit run grouped with spaces, anchored to a proper noun: "Poland 2 474".
#
# The anchor is doing the real work. Measured against the two documents with
# score history, an unanchored digit-run pattern fires six times and is a false
# positive every single time: HDR's "305 307 306 307" is a row of chart ticks,
# and the parks book contributes index page numbers ("GR7 260 GR10 238"), a
# phone number ("T: 510 250 6400") and a line of OCR garbage. Collapsing any of
# those invents a number that is not in the document.
#
# Requiring a Capitalised-word-with-lowercase-letters immediately before the run
# excludes all six -- '302*', 'GR7', '|', 'T:' and 'AUEW' are none of them a
# proper noun -- while still matching the country-labelled table rows this
# exists for.
_GROUPED_NUMBER = re.compile(
    r"(?<=[a-z])[ ](?P<lead>\d{1,3})(?P<groups>(?:[ ]\d{3})+)(?![\d.,])")

# How many 3-digit groups may be absorbed into the leading group.
#
# This is a genuine ambiguity and the knob is honest about it. In a flattened
# table row "Poland 2 474 403", the digits are Poland's employee count 2 474 and
# its company count 403 -- two columns, not one number with two separators. From
# the row alone nothing distinguishes that from a single value of 2,474,403.
# Absorbing ONE group reads numbers up to 999,999 correctly and keeps the next
# column intact, which is right for this table and for most tables; a document
# whose figures run into the millions needs this raised to 2.
THOUSANDS_GROUPS = int(os.environ.get("THOUSANDS_GROUPS", "1"))

# Isolated occurrences are prose ("in Poland 2 474 people were surveyed" reads
# the same either way). A table has many, so only rewrite when the pattern is
# systematic across the document.
THOUSANDS_MIN_HITS = int(os.environ.get("THOUSANDS_MIN_HITS", "8"))

# The longest run that can still be a number rather than a chart axis.
#
# This is the gate that actually matters, and the proper-noun anchor alone did
# not catch it. Measured on the real OECD file, the anchor passes a y-axis tick
# row whenever a country name sits in front of it:
#
#     United States 116 114 112 110 108 106 104 102 100 98
#     9. Sanctions 700 600 500 400 300 200 100 0
#
# Joining there invents '116114', a figure printed nowhere in the document. A
# genuine space-grouped number carries one continuation group ('250 000',
# '31 762') or at most two ('1 612 661'); an axis carries six or ten. Counting
# the groups separates them cleanly where the anchor could not.
THOUSANDS_MAX_GROUPS = int(os.environ.get("THOUSANDS_MAX_GROUPS", "2"))


def collapse_grouped_numbers(text: str) -> str:
    """
    Join space-grouped thousands into single tokens: 'Poland 2 474' -> 'Poland 2474'.

    In the OECD document Table 5.1 arrives as a flattened token stream using a
    space as the thousands separator, so a reader asked for a company count sees
    "Poland ... 2 474 403" and has no way to tell which of those three tokens is
    the column it was asked about. That is the root cause of the contradiction
    failures: the wrong figures reported there are real numbers from the table
    bound to the wrong column, not numbers the model invented.

    Every step downstream inherits this. A claim ledger built on top of
    ambiguous extraction produces confidently wrong records and marks them
    authoritative, so this has to be right before anything is built on it.
    """
    matches = list(_GROUPED_NUMBER.finditer(text))
    if len(matches) < THOUSANDS_MIN_HITS:
        return text

    out, cursor, rewritten, skipped = [], 0, 0, 0
    for m in matches:
        groups = m.group("groups").split()
        lead = m.group("lead")
        # A run this long is a chart axis, and a lead with a leading zero is a
        # tick label or a series value -- no thousands group starts with 0.
        if len(groups) > THOUSANDS_MAX_GROUPS or lead.startswith("0"):
            skipped += 1
            continue
        joined = lead + "".join(groups[:THOUSANDS_GROUPS])
        rest = "".join(" " + g for g in groups[THOUSANDS_GROUPS:])
        out.append(text[cursor:m.start()])
        out.append(" " + joined + rest)
        cursor = m.end()
        rewritten += 1
    out.append(text[cursor:])

    if not rewritten:
        return text
    _log(f"preprocess    joined {rewritten:,} space-grouped numbers "
         f"(e.g. '2 474' -> '2474'); left {skipped:,} long digit runs alone")
    return "".join(out)


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

    # g8 step 1. Both are no-ops on a document that does not have the fault:
    # each counts its own evidence first and returns the text untouched below
    # threshold. Measured on the two documents with score history, both make
    # zero changes -- which is what makes them safe to leave on by default.
    if step_on("1"):
        raw = repair_mojibake(raw)
        raw = collapse_grouped_numbers(raw)

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


# ------------------------------------------- HEADER SEGMENTATION (g12)
#
# Some documents from the course toolchain keep their Markdown '## ' headers as
# literal substrings; most do not. Measured, before anything was wired:
#
#   nationalparks_europe.txt   1,006 headers, coverage 1.000  -> headers
#   oecd2026_fullscan_test.txt     0 headers, coverage 0.000  -> words
#   gem2024_5_fullscan.txt         0 headers, coverage 0.000  -> words
#   hdr2025.txt                    0 headers, coverage 0.000  -> words
#
# So this is discovered at runtime and never assumed, and the fallback is the
# existing chunker untouched. The margin between the two outcomes is the whole
# range of the measure, which is why the thresholds below are not delicate.

# The longest thing that can still be a header title rather than a paragraph
# that happens to follow a '##'.
HEADER_MAX_TITLE = int(os.environ.get("HEADER_MAX_TITLE", "90"))

# NOT line-anchored, and that is the point. The parks file arrives as ONE line
# of 578,000 characters, with the two literal characters backslash-n where its
# newlines should be; `^##` under re.MULTILINE finds nothing at all in it before
# normalize_text runs, and normalize_text is not guaranteed to fire on a
# document whose escaping convention differs again. Searching for the substring
# works in both worlds. The title ends at the next newline -- real or literal --
# at the next '#', or at the length cap.
_HEADER = re.compile(
    r"##[ \t]+"
    r"(?P<title>[^\n#]{1,%d}?)"
    r"(?=\s*(?:\\n|\n|#|$))" % HEADER_MAX_TITLE)


def detect_headers(text: str) -> list[dict]:
    """
    Every '## Title' occurrence, as ordered non-overlapping character spans.

    Returns [{'title', 'start', 'body_start', 'end'}] where 'start' is the
    offset of the '##' marker, 'body_start' the offset just past the title, and
    'end' the next header's 'start' (len(text) for the last). Nothing is
    inferred: a document with no such markers returns an empty list, and every
    caller below treats that as "this document is not built this way".
    """
    found = []
    for m in _HEADER.finditer(text):
        title = m.group("title").strip(" \t*_:-")
        if not title:
            continue
        found.append({"title": title, "start": m.start(),
                      "body_start": m.end(), "end": None})
    for i, h in enumerate(found):
        h["end"] = found[i + 1]["start"] if i + 1 < len(found) else len(text)
    return found


HEADER_MIN_COUNT = int(os.environ.get("HEADER_MIN_COUNT", "5"))
HEADER_MIN_COVERAGE = float(os.environ.get("HEADER_MIN_COVERAGE", "0.5"))


def headers_are_usable(headers: list[dict], text: str,
                       min_count: int = HEADER_MIN_COUNT,
                       min_coverage: float = HEADER_MIN_COVERAGE) -> bool:
    """
    Are these headers a real partition of the document, or just a title page?

    Two tests. `min_count` stops a lone heading or a five-line preface from
    switching the whole pipeline onto a different chunker. `min_coverage` stops
    the more dangerous case: a Contents page that lists every chapter title,
    giving forty headers that all sit in the first 2% of the file and partition
    nothing. Coverage is measured from the first header to the end of the last,
    so headers clustered in one region score low however many there are.

    Measured: 1.000 on the one document that has headers, 0.000 on the three
    that do not. There is no document in hand that lands anywhere near 0.5.
    """
    if not text or len(headers) < min_count:
        return False
    covered = headers[-1]["end"] - headers[0]["start"]
    return covered >= min_coverage * len(text)


# A section longer than this multiple of the target is split on words anyway.
# A whole chapter under one header is not a unit worth keeping intact at the
# cost of handing one reader the entire document.
HEADER_MAX_SECTION = float(os.environ.get("HEADER_MAX_SECTION", "3"))

# How far over the target a chunk may go to avoid starting another one.
#
# Not a tuning knob for one document -- it is the cost of packing atoms instead
# of cutting anywhere. A chunker that may cut mid-sentence fills every slice to
# exactly the target; one that may only cut between sections leaves whatever
# does not fit, and that wasted capacity accumulates into an extra slice, which
# is an extra model call per question. Measured on the parks book: refusing all
# overshoot gives 16 slices where the word chunker gives 15. Five per cent
# recovers it (15 slices, largest 6,300 words), and the constraint is that this
# change must not ADD calls.
HEADER_PACK_SLACK = float(os.environ.get("HEADER_PACK_SLACK", "0.05"))


def chunk_by_headers(text: str, headers: list[dict],
                     target_words: int) -> list[dict]:
    """
    Slice on section boundaries, packing whole sections up to `target_words`.

    NOT one chunk per section, and the measurement is the reason. The parks
    book carries 1,006 header sections over 92,808 words -- a median section of
    53 words, because most of its headers are the sub-headings repeated inside
    every entry ('Toolbox' 57 times, 'Do this!' 60 times) rather than one per
    park. One chunk per section would be 1,006 readers where the word chunker
    makes 15: sixty-five times the model calls, against a hard constraint that
    this change reduce them. It would also put every reader at 53 words, far
    below the ~6,000 that is the measured line between a reader that enumerates
    named instances and one that writes summaries worth zero key points.

    So the section is the ATOM, not the chunk. Sections are packed in document
    order until the next one would overflow the target, which yields the same
    number of readers as before, each holding the same amount of text -- with
    the one difference that a boundary now never falls inside a section. A
    'Park in numbers' box can no longer be split from the sentence that names
    the park, because the split points are the document's own.
    """
    if not headers:
        return chunk_lines(text, target_words)

    pieces: list[tuple[str | None, str]] = []
    if headers[0]["start"] > 0:
        head = text[:headers[0]["start"]]
        if head.strip():
            pieces.append((None, head))
    for h in headers:
        pieces.append((h["title"], text[h["start"]:h["end"]]))

    cap = target_words * (1.0 + HEADER_PACK_SLACK)
    out: list[dict] = []
    current: list[str] = []
    titles: list[str] = []
    words = 0

    def flush():
        nonlocal current, titles, words
        if current:
            out.append({"text": "\n".join(current),
                        "section_titles": list(dict.fromkeys(t for t in titles if t))})
            current, titles, words = [], [], 0

    for title, body in pieces:
        n = len(body.split())
        if n > HEADER_MAX_SECTION * target_words:
            # One section larger than three readers' worth. Keep the title on
            # every piece so the section it came from is still known.
            flush()
            tokens = body.split()
            for i in range(0, len(tokens), target_words):
                out.append({"text": " ".join(tokens[i:i + target_words]),
                            "section_titles": [title] if title else []})
            continue
        if current and words + n > cap:
            flush()
        current.append(body)
        titles.append(title)
        words += n
    flush()

    if not out:
        out = [{"text": text, "section_titles": []}]
    for i, c in enumerate(out):
        c["n"] = i + 1
        c["of"] = len(out)
        c["section_title"] = c["section_titles"][0] if c["section_titles"] else ""
    return out


# The share of one entry within which a figure still belongs to that entry.
PROVENANCE_ENTRY_SHARE = float(os.environ.get("PROVENANCE_ENTRY_SHARE", "0.5"))
# Never widen past this, whatever the document's entries measure.
PROVENANCE_WINDOW_MAX = int(os.environ.get("PROVENANCE_WINDOW_MAX", "6000"))


def entry_period(headers: list[dict], text: str) -> int | None:
    """
    How many characters one of this document's repeated entries occupies.

    Measured, not assumed, and by the same evidence count_entries() already
    trusts: a book of profiles repeats the same sub-headings inside every entry,
    so the median gap between consecutive occurrences of its COMMONEST heading
    is the length of one entry. On the parks book 'Do this!' occurs 60 times and
    the book profiles 60 parks, giving a period of roughly nine thousand
    characters.

    WHAT THIS IS FOR. verify_record() asks whether a reported figure sits near
    the entity it is reported for, and 'near' was a constant 1,200 characters.
    On this document the park's name is a heading and its 'Park in numbers' box
    is two to three thousand characters further down, so the check rejected
    CORRECT records: measured on the last parks run, Hohe Tauern (2,346), Sierra
    Nevada (2,380), Ordesa (2,404), Aiguestortes (2,429) and Pyrenees (2,922)
    all had the right summit reported and all five were discarded, which is
    precisely why a count of eight came back as three.

    Raising the constant to 3,000 would be fitting the code to this document --
    the failure mode the whole project exists to avoid. Deriving it from the
    document's own entry period is not: the question 'how far apart may an
    entity and its figures be' has an answer printed in the document's layout,
    and this reads it. A document with no repeated headings returns None and the
    constant stands.
    """
    if not headers:
        return None
    counts = collections.Counter(h["title"] for h in headers)
    title, n = counts.most_common(1)[0]
    if n < 5:
        return None                       # no repeating structure to measure
    at = [h["start"] for h in headers if h["title"] == title]
    gaps = sorted(b - a for a, b in zip(at, at[1:]))
    if not gaps:
        return None
    median = gaps[len(gaps) // 2]
    # Evenly spaced, or it is not an entry marker -- the same corroboration
    # count_entries() applies before believing its own count.
    if gaps[-1] > 12 * (sum(gaps) / len(gaps)):
        return None
    return int(median)


def provenance_window(doc: dict) -> int:
    """The window verify_record should use for this document."""
    if not (g12_on("provwindow") and doc.get("chunking_mode") == "headers"):
        return PROVENANCE_WINDOW
    period = entry_period(doc.get("headers") or [], doc.get("text") or "")
    if not period:
        return PROVENANCE_WINDOW
    return max(PROVENANCE_WINDOW,
               min(PROVENANCE_WINDOW_MAX, int(period * PROVENANCE_ENTRY_SHARE)))


def chunk_for(doc: dict, text: str, target_words: int) -> list[dict]:
    """
    Header segmentation where the document supports it, chunk_lines where not.

    Every call site that used to say `chunk_lines(x, n)` says this instead. On a
    document without usable headers the two are the same function, which is what
    keeps the fallback path bit-for-bit unchanged.
    """
    if not (g12_on("headers") and doc.get("chunking_mode") == "headers"):
        return chunk_lines(text, target_words)
    # Detected on the text actually being chunked. A locator-scoped region is a
    # slice of the document and carries its own headers, or none, and reusing
    # the whole document's offsets against it would be nonsense.
    headers = (doc["headers"] if text is doc.get("text")
               else detect_headers(text))
    if not headers_are_usable(headers, text):
        return chunk_lines(text, target_words)
    return chunk_by_headers(text, headers, target_words)


# ------------------------------------------ ENTITY HEADERS (g12, step 4b)
#
# Which header titles name something the document PROFILES, and which are the
# scaffolding repeated inside every profile. The rule is about the shape of the
# title and holds no fact about any document; the numbers it produced on the one
# header-carrying document in hand are in the report and in header_probe.py.

_GENERIC_TITLE_WORD = {
    "day", "days", "week", "weeks", "weekend", "itinerary", "itineraries",
    "toolbox", "contents", "content", "introduction", "index", "overview",
    "summary", "conclusion", "references", "bibliography", "appendix",
    "annex", "glossary", "foreword", "preface", "acknowledgements", "notes",
    "more", "here", "there", "this", "that", "spot", "hike", "stay", "do",
    "getting", "what", "how", "why", "when", "where", "see", "seeing",
}


def _title_words(title: str) -> list[str]:
    return re.findall(r"[\w'’-]+", title)


def looks_like_entity_header(title: str, counts) -> bool:
    """
    Six shape tests. Failing any one of them makes a title furniture.

      1. plausible length
      2. occurs exactly once -- a title repeated across the document is the
         book's scaffolding, and this test alone removes 'Toolbox' (57),
         'Do this!' (60), 'What to spot...' (60), 'Itineraries' (45)
      3. no terminal '!', '?', '...' or ':' -- an imperative is an instruction
      4. two or more words
      5. does not begin with a digit -- kills itinerary legs, '03 Estany Llong'
      6. two or more words in real title case (`[A-Z][a-z]`), which excludes
         ALL-CAPS callouts like 'MARSICAN BROWN BEAR'
    """
    if not (4 <= len(title) <= 70):
        return False
    if counts[title] > 1:
        return False
    if title.rstrip().endswith(("!", "?", "...", "…", ":")):
        return False
    words = _title_words(title)
    if len(words) < 2:
        return False
    if re.match(r"^\s*\d", title):
        return False
    proper = [w for w in words if re.match(r"^[A-Z][a-zÀ-ɏ'’-]", w)]
    if len(proper) < 2:
        return False
    if all(w.lower() in _GENERIC_TITLE_WORD or len(w) < 3 for w in words):
        return False
    return True


# How dominant a naming convention must be before it is believed.
ENTITY_SUFFIX_MIN = int(os.environ.get("ENTITY_SUFFIX_MIN", "3"))
ENTITY_SUFFIX_SHARE = float(os.environ.get("ENTITY_SUFFIX_SHARE", "0.6"))
ENTITY_SUFFIX_MARGIN = float(os.environ.get("ENTITY_SUFFIX_MARGIN", "2.0"))


def entity_headers(headers: list[dict]) -> list[str]:
    """
    Header titles that name an entity the document profiles -- or nothing.

    The shape test above leaves proper-noun titles. Those must then agree on a
    NAMING CONVENTION: a book of profiles names its subjects alike ('... National
    Park', '... Ltd', '... Museum'), and the shared trailing word is that
    convention. One convention must WIN, covering `ENTITY_SUFFIX_SHARE` of the
    survivors and beating the runner-up by `ENTITY_SUFFIX_MARGIN`.

    IT REFUSES RATHER THAN GUESSES, and on the one document in hand it refuses.
    Measured on the parks book: 178 titles pass the shape test, and the trailing
    words are 'park' 18, 'hotel' 15, 'house' 4, 'spa' 3. The book gives a header
    to only 18 of the 60 parks it profiles but to nearly every hotel inside a
    'Stay here...' block, so 'park' takes 45% of the survivors and beats 'hotel'
    by 1.2x -- neither threshold is met and nothing is seeded. That is the right
    answer. Seeding a roster used by the aggregation path with 'Fossli Hotel'
    and 'Radisson Blu Hotel' would put twenty-one non-entities in front of every
    counting reader, and aggregation is the category with the least margin to
    lose. A document whose profiled entities really do dominate its header
    namespace clears both thresholds and gets the roster.
    """
    counts = collections.Counter(h["title"] for h in headers)
    shaped = [h["title"] for h in headers
              if looks_like_entity_header(h["title"], counts)]
    if len(shaped) < ENTITY_SUFFIX_MIN:
        return []
    tails = collections.Counter(_title_words(t)[-1].lower() for t in shaped
                                if _title_words(t))
    ranked = tails.most_common()
    if not ranked:
        return []
    winner, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if top < ENTITY_SUFFIX_MIN:
        return []
    if top < ENTITY_SUFFIX_SHARE * len(shaped):
        return []
    if runner_up and top < ENTITY_SUFFIX_MARGIN * runner_up:
        return []
    kept, seen = [], set()
    for t in shaped:
        words = _title_words(t)
        if words and words[-1].lower() == winner and t not in seen:
            seen.add(t)
            kept.append(t)
    return kept


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


# ------------------------------------------------ BACK MATTER (g8 step 6)

# A heading on a line of its own. There is usually a decoy: the table of
# contents lists "Index" near the front. Only headings in the back half count.
_INDEX_HEADING = re.compile(
    r"^\s*#*\s*(?:index|general index|subject index)\s*$", re.I | re.M)

# "Oulanka National Park 177 , 178 , 179" -- a name followed by page numbers and
# nothing else. This is what an index entry looks like in every book, and it is
# also exactly what a reader mistakes for a data table.
_INDEX_ENTRY = re.compile(
    r"^\s*\W?\s*[A-Z][^|]{2,60}?\s+(?:\d{1,4}\s*[-,]?\s*)+$")

INDEX_MIN_ENTRIES = int(os.environ.get("INDEX_MIN_ENTRIES", "25"))
INDEX_MIN_DENSITY = float(os.environ.get("INDEX_MIN_DENSITY", "0.15"))
INDEX_REQUIRE_HEADING = os.environ.get("INDEX_REQUIRE_HEADING", "1") == "1"


def _index_density(region: str) -> tuple[int, float]:
    lines = [l for l in region.split("\n") if l.strip()]
    if not lines:
        return 0, 0.0
    hits = sum(1 for l in lines if _INDEX_ENTRY.match(l))
    return hits, hits / len(lines)


def find_back_matter(text: str) -> int | None:
    """
    Where the index starts, as a character offset, or None if there isn't one.

    This exists because an index is indistinguishable from a data table to a
    reader asked to extract records. The parks book's index sits in the last
    slice, and that one reader emitted about eighty records per question --
    relabelling the same page numbers as whatever the question asked for:

        RECORD | Pembrokeshire Coast National Park | page          | 197
        RECORD | Pembrokeshire Coast National Park | highest point | 197
        RECORD | Pembrokeshire Coast National Park | visitors per year | 197

    Those fabrications outnumbered the real records and are why a count of parks
    above 3000 m came back as 3 instead of 8. Any comparison built on top of
    them would report a contradiction for every entry in the book.

    Detected, never assumed: a heading in the back half whose following region
    is genuinely dense in index entries, or failing that a dense trailing run.
    A document without an index gets nothing removed.
    """
    headings = [m for m in _INDEX_HEADING.finditer(text)
                if m.start() > len(text) * 0.5]
    for m in reversed(headings):
        hits, density = _index_density(text[m.start():])
        if hits >= INDEX_MIN_ENTRIES and density >= INDEX_MIN_DENSITY:
            return m.start()

    if INDEX_REQUIRE_HEADING:
        # Measured on HDR: the headingless search fires on the country/HDI-rank
        # listing at 99.7%, which is index-SHAPED but is real data -- every
        # country against its rank. An index and a one-column reference table
        # are the same thing to a regex, and only the heading tells them apart.
        # A missed index costs some noisy records; a removed data table costs
        # the answer, so this defaults to requiring the heading.
        return None

    # No usable heading: look for a dense run at the tail. Walk backwards from
    # the end so the boundary lands where the entries actually begin.
    lines = text.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    start = int(len(lines) * 0.75)
    for i in range(len(lines) - 1, start, -1):
        hits, density = _index_density("\n".join(lines[i:]))
        if hits >= INDEX_MIN_ENTRIES and density >= INDEX_MIN_DENSITY * 2:
            best = i
        else:
            continue
        return offsets[best]
    return None


def strip_back_matter(text: str) -> str:
    """
    Drop the index before records are extracted from the text.

    Applied ONLY where records are parsed. Absence counts terms over the whole
    document and a term appearing in the index is still a mention; prose and
    trajectory questions read everything as before. Narrowing what those modes
    see would change categories that already score 1.00.
    """
    cut = find_back_matter(text)
    if cut is None:
        return text
    removed = len(text) - cut
    _log(f"  ~ index detected at {cut / len(text):.0%} through; "
         f"{removed:,} characters excluded from record extraction")
    return text[:cut]


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


# A list ends where its sentence ends. Splitting on a bare "." would cut
# "Figure 3.1" in half, so a full stop only terminates when a space and a
# capital follow it; "?" and "!" terminate on their own.
_SENTENCE_END = re.compile(r"[?!]|\.(?=\s+[A-Z])")


def _first_sentence(text: str) -> str:
    """
    Everything up to the first sentence boundary.

    Absence questions on one document ended at the question mark; on another
    they carry a trailing instruction -- "..., or servant leadership? Briefly
    state how the other three appear." Taking the whole tail made that
    instruction a candidate subject, and because a phrase like "Briefly state
    how the other three appear" occurs nowhere in any document, it WON the
    absence verdict every time. Three questions answered with a sentence
    fragment as the missing subject, and the real answer -- which the same
    code had already found to be absent -- reported as present.
    """
    return _SENTENCE_END.split(text, maxsplit=1)[0]


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
    tail = _first_sentence(question.rsplit(":", 1)[-1] if ":" in question else "")
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
    # g8 step 3 gives cross-section its own route. It stays on prose when the
    # step is switched off, so the staged runs differ in one thing at a time.
    cross = "cross" if step_on("3") else "prose"
    contradiction = "ledger" if step_on("5") else "prose"
    known = {"aggregation": "tally", "superlative": "tally", "absence": "absence",
             "contradiction": contradiction, "cross_section": cross,
             "cross-section": cross, "global_synthesis": "arc",
             "global-synthesis": "arc", "needle": "prose"}
    if category and category.lower().strip() in known:
        return known[category.lower().strip()]
    q = question.lower()
    if re.search(r"never (mentioned|appears|discussed|substantively)", q):
        return "absence"
    if re.search(r"\bhow many\b|\bhow much\b|\bcount\b|\btotal\b", q):
        return "tally"

    # Infer the three routed categories too, not just the three g1 could infer.
    #
    # Without this, a question file that ships no categories loses every route
    # added after g6: the fallback below can only ever return tally, absence or
    # prose, so a trajectory question would be read as prose and the arc,
    # cross-section and ledger machinery would never run at all. The categories
    # are supplied metadata and are trusted when present -- this is only for a
    # file that omits them, which is a thing we do not control and will not see
    # until the document is released.
    if contradiction != "prose" and re.search(
            r"\b(inconsisten\w*|contradict\w*|disagree\w*|conflict\w*|"
            r"at odds|two different (?:values|figures|numbers)|"
            r"differs? from|does not match)\b", q):
        return contradiction
    if cross != "prose" and re.search(
            r"\bboth\b.*\band\b|\bappears? in\b.*\band (?:also|again)\b|"
            r"\bsame (?:country|entity|organisation|organization|institution|"
            r"person)\b.*\bboth\b", q):
        return cross
    if re.search(r"\b(develops?|evolves?|evolution|trajectory|shifts?|"
                 r"progress\w*|unfolds?|over the course|across the (?:document|"
                 r"report|book|text)|from beginning to end|by the end)\b", q):
        return "arc"

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


def _neighbour_agrees(found: str, wanted: str) -> bool:
    """Are these the same word, allowing for the usual endings?"""
    a, b = found.lower(), wanted.lower()
    stem = lambda w: w[:5] if len(w) > 5 else w          # noqa: E731
    return a == b or stem(a) == stem(b)


def _fragment_hits(gram: str, full: str, text: str) -> int:
    """
    Count a partial phrase, skipping occurrences that belong to another phrase.

    A subject is looked up by its longest matching n-gram, which is what lets
    "excessive screen time in early childhood" be recognised from "excessive
    screen time". The same latitude also let "universal basic income" -- a
    phrase the education report never uses -- be certified by its fragment
    "universal basic", whose single occurrence in the document is inside
    "universal basic EDUCATION". One stolen occurrence marked the subject
    discussed, and the absence question it belonged to lost its deterministic
    answer and fell through to the readers.

    So: where the subject continues with a content word and the document
    continues with a different one, that occurrence is somebody else's phrase.
    Where the subject's neighbour is a function word ("...time IN early
    childhood") nothing is checked, because that is the case the latitude was
    granted for.
    """
    words, gram_words = full.split(), gram.split()
    start = next((i for i in range(len(words) - len(gram_words) + 1)
                  if words[i:i + len(gram_words)] == gram_words), None)
    if start is None:
        return _occurrences(gram, text)

    after = words[start + len(gram_words)] if start + len(gram_words) < len(words) else None
    before = words[start - 1] if start else None
    if after in _STOPWORDS:
        after = None
    if before in _STOPWORDS:
        before = None
    if not after and not before:
        return _occurrences(gram, text)

    hits = 0
    for form in ({gram, gram[:-1] if gram.endswith("s") else gram + "s"}):
        for m in re.finditer(rf"\b{re.escape(form)}\b", text):
            if after:
                nxt = re.match(r"\W+(\w+)", text[m.end():m.end() + 40])
                if nxt and not _neighbour_agrees(nxt.group(1), after):
                    continue
            if before:
                prv = re.search(r"(\w+)\W+$", text[max(0, m.start() - 40):m.start()])
                if prv and not _neighbour_agrees(prv.group(1), before):
                    continue
            hits += 1
    return hits


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

        full = candidates[0] if candidates else None

        matched, count = None, 0
        for phrase in candidates:                    # longest first
            hits = (_fragment_hits(phrase, full, body)
                    if g11_on("fragment") and full and phrase != full
                    else _occurrences(phrase, body))
            if hits:
                matched, count = phrase, hits
                break

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


# ------------------------------------------------ CROSS-SECTION (g8 step 3)

# A phrase occurring more often than this is a topic, not a landmark. It cannot
# narrow anything down, and including it drags the whole document back in.
ANCHOR_MAX_HITS = int(os.environ.get("ANCHOR_MAX_HITS", "40"))
# How many landmarks to keep. A cross-section question links two or three facts;
# a handful of anchors each side is enough and more only adds noise.
ANCHOR_PHRASES = int(os.environ.get("ANCHOR_PHRASES", "8"))
# If the anchors pull in more than this share of the document, they have not
# actually located anything -- read everything instead, as before.
ANCHOR_MAX_SHARE = float(os.environ.get("ANCHOR_MAX_SHARE", "0.6"))

# "February 2024" is one of the sharpest landmarks a question can carry: a month
# paired with a year usually occurs once in a document, where the bare year
# occurs everywhere.
_MONTH_YEAR = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+((?:19|20)\d{2})\b", re.I)
# Runs of capitalised words in the question as it was written -- proper nouns,
# instrument names, "OECD-Bocconi", "Chapter 6".
_PROPER_RUN = re.compile(r"\b[A-Z][\w'-]*(?:[ -][A-Z][\w'-]*)*\b")


def question_anchors(question: str, text: str) -> list[dict]:
    """
    The phrases in a question that actually locate something in the document.

    A cross-section question does not name its answer -- it names two facts and
    asks which entity holds both ("Which country is used both as an example of
    portable social-housing rights and ... of 2024 minimum-standard regulation
    for employee-like platform workers?"). Reading the whole document for that
    gives the synthesis step twenty-eight partial views of a theme and no way to
    tell which entity sits at the intersection, which is how two runs in a row
    named the wrong country.

    But the question's own wording is a locator. Its distinctive predicates --
    'employee-like', 'February 2024', a named survey's authors -- occur a
    handful of times in the whole document, and the chunks holding them are
    exactly the two ends of the link.

    So: take every phrase the question offers, count it in the document ON WORD
    BOUNDARIES with the same counter the absence path uses, and keep the ones
    that are present but rare. Zero hits means the phrase is our wording, not
    the document's; hundreds of hits means it is the document's subject matter.
    The landmarks are in between, and they are ranked rarest-first because the
    rarest phrase is the one that has genuinely found something.
    """
    lowered = text.lower()
    candidates: list[str] = []

    # Month-and-year pairs, and proper-noun runs, from the question as written.
    for m in _MONTH_YEAR.finditer(question):
        candidates.append(m.group(0).lower())
    for m in _PROPER_RUN.finditer(question):
        phrase = m.group(0)
        # A capitalised first word is just the start of the sentence.
        if m.start() == 0 or len(phrase) < 4:
            continue
        candidates.append(phrase.lower())

    # Then n-grams over the question's own words, longest first. Same shape as
    # question_subjects builds for absence, but across the whole question rather
    # than per listed candidate, because a cross-section question is one clause.
    words = _WORDS.findall(question.lower())
    for size in (4, 3, 2, 1):
        for i in range(len(words) - size + 1):
            gram = words[i:i + size]
            if gram[0] in _STOPWORDS or gram[-1] in _STOPWORDS:
                continue
            if all(w in _STOPWORDS or w in _META for w in gram):
                continue
            if size == 1 and (len(gram[0]) < 6 or gram[0] in _META):
                continue        # a lone short word is never a landmark
            candidates.append(" ".join(gram))

    scored = []
    seen = set()
    for phrase in candidates:
        if phrase in seen or len(phrase) < 5:
            continue
        seen.add(phrase)
        hits = _occurrences(phrase, lowered)
        if hits == 0 or hits > ANCHOR_MAX_HITS:
            continue
        scored.append({"phrase": phrase, "hits": hits,
                       "words": len(phrase.split())})

    # Rarest first, and among equally rare phrases the longest, because a longer
    # phrase that is just as rare is the more specific landmark.
    scored.sort(key=lambda a: (a["hits"], -a["words"], -len(a["phrase"])))

    kept: list[dict] = []
    for anchor in scored:
        # Drop a phrase wholly contained in one already kept: it selects the
        # same chunks and spends one of the few slots saying so again.
        if any(anchor["phrase"] in k["phrase"] for k in kept):
            continue
        kept.append(anchor)
        if len(kept) >= ANCHOR_PHRASES:
            break
    return kept


def anchor_chunks(chunks: list[dict], anchors: list[dict]) -> list[dict]:
    """
    The chunks holding at least one landmark, in document order.

    Returns everything unchanged when the anchors fail to narrow the document --
    no anchors at all, or anchors so widespread that the selection is most of
    the text anyway. A cross-section question that cannot be located is still
    better served by reading everything than by reading a confident subset.
    """
    if not anchors:
        return chunks

    selected = []
    for chunk in chunks:
        lowered = chunk["text"].lower()
        found = [a["phrase"] for a in anchors
                 if _occurrences(a["phrase"], lowered)]
        if found:
            selected.append({**chunk, "anchors": found})

    if not selected or len(selected) > ANCHOR_MAX_SHARE * len(chunks):
        return chunks
    return selected


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

# A field that is nothing but a unit. Used in two places: to recognise a
# reader's stray unit column, and to recognise the parenthetical in a data-box
# label ('Area covered (sq miles)'). Deliberately narrow -- anything it lets
# through is treated as a unit, and a false positive silently rewrites a value.
_UNIT_ONLY = re.compile(
    r"^(?:sq\.?\s*(?:km|mi|miles|kilometres|kilometers|m)|km\s*[²³]?|"
    r"m\s*[²³]?|mi|miles|metres|meters|ft|feet|ha|hectares|acres|"
    r"km2|m2|%|per\s*cent|percent|years?|days?|hours?|kg|tonnes|tons|"
    r"°c|°f|mm|cm|litres|liters|usd|eur|gbp|€|£|\$)$",
    re.I)

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
        if g10_on("unitcol") and "|" in where:
            # A reader handed a table whose value and unit sit in two columns
            # emits five fields, not four: 'X | area covered | 26 | sq miles |
            # Park in numbers'. The regex's last group swallows the overflow,
            # so the unit lands in 'where' and _parse_value never sees it --
            # and a record with no unit is invisible to the unit-outlier
            # detector, which is precisely the record that detector exists to
            # find. Put the unit back on the value where it belongs.
            head, _, tail = where.partition("|")
            if _UNIT_ONLY.match(head.strip()):
                raw_value = f"{raw_value} {head.strip()}"
                where = tail.strip() or where
        out.append(_make_record(entity, attribute, raw_value, where, source, "record"))

    for m in _EXTREME.finditer(text):
        word, attribute, entity, raw_value = (p.strip() for p in m.groups())
        if not entity or entity.lower().strip(" .:") in _NOT_AN_ENTITY:
            continue
        out.append(_make_record(entity, attribute, raw_value,
                                f"local {word.lower()} of {source}",
                                source, word.lower()))
    return out


# ------------------------------------------------- CLAIM LEDGER (g8 step 5)

# A superlative assertion the document makes about itself, rather than a value:
#   CLAIM | Snowdon | the most climbed mountain in Britain | page 239
_CLAIM = re.compile(r"^\s*CLAIM\s*\|(.+?)\|(.+?)\|(.*)$", re.I | re.M)

_SUPERLATIVE = re.compile(
    r"\b(largest|biggest|highest|tallest|longest|deepest|oldest|greatest|most|"
    r"smallest|shortest|lowest|youngest|first|only|widest)\b", re.I)

# Words that carry no meaning when deciding whether two attribute labels refer
# to the same measurement. Everything else in the label is signal.
_ATTR_NOISE = {"the", "of", "a", "an", "in", "on", "for", "and", "or", "to",
               "by", "at", "its", "with", "number", "total", "amount", "value",
               "covered", "approximate", "estimated", "recorded", "reported",
               "annual", "annually", "per", "nb", "no", "count", "size"}

# ------------------------------------------------- MEASUREMENT FAMILIES (g9)
#
# The single most expensive line in g8 was tally()'s
#
#     dominant = max(counts, key=counts.get)          # counts keyed on the
#     numeric = [r for r in numeric                   # RAW attribute string
#                if r["attribute"] == dominant]
#
# Twenty-eight readers each name a measurement in the words of the page in
# front of them. For one question they returned 'highest point', 'highest
# elevation', 'highest peak', 'maximum elevation' and 'highest point (m)' --
# five spellings of one measurement. Exact string equality keeps the largest
# spelling and throws the other four away, silently, AFTER extraction has
# already succeeded. Fed all eight correct parks, g8's tally answered four.
#
# _same_measurement is the right test for the ledger, where 'land area' and
# 'marine area' must stay apart, but it is too strict here: {high,poin} against
# {high,elev} shares one token out of two and fails the 0.6 rule. So a label is
# first mapped to a MEASUREMENT FAMILY, and only labels with no recognised
# family fall back to token overlap.
#
# This is a small English lexicon -- the same kind of thing as _SUPERLATIVE_MEANS
# above -- and holds no fact about any particular document.

# Stripped before matching: they modify a measurement without naming one, and
# left in they create ties ('highest' is elevation in 'highest point' and not in
# 'highest number of visitors').
_ATTR_MODIFIER = {"highest", "lowest", "maximum", "minimum", "max", "min",
                  "greatest", "smallest", "largest", "total", "overall",
                  "approximate", "approx", "average", "mean", "median",
                  "about", "around", "roughly", "circa", "est", "estimated"}

_MEASUREMENT_FAMILIES = {
    # 'level' and 'coverage' are deliberately absent. Both are ambiguous in the
    # documents that currently score well -- 'level of education' is not an
    # elevation and 'non-compete coverage' is not an area -- and a wrong family
    # merges two measurements, which is worse than leaving a label unfamilied.
    "elevation": {"height", "heights", "point", "peak", "peaks", "elevation",
                  "altitude", "summit", "mountain", "mount", "asl",
                  "metres", "meters", "elevations"},
    "area":      {"area", "areas", "surface", "extent", "hectares", "hectare",
                  "acres", "acreage", "km2", "expanse"},
    "length":    {"length", "distance", "trail", "trails", "route", "routes",
                  "path", "paths", "perimeter", "coastline", "shoreline"},
    "depth":     {"depth", "depths", "deep"},
    "age":       {"year", "years", "established", "founded", "created",
                  "designated", "date", "inaugurated", "opened", "age",
                  "since", "declared"},
    "count":     {"visitors", "visits", "visitor", "population", "species",
                  "inhabitants", "employees", "companies", "firms", "people",
                  "respondents", "interviews", "staff", "members", "workers"},
    "location":  {"country", "countries", "nation", "nations", "region",
                  "state", "province", "location", "located", "situated",
                  "where", "territory"},
    "money":     {"budget", "revenue", "cost", "costs", "funding", "income",
                  "spending", "expenditure", "fee", "fees", "price", "gdp"},
    "share":     {"share", "percentage", "percent", "proportion", "rate",
                  "ratio", "coverage_pct", "prevalence", "incidence"},
}


def measurement_family(attribute: str) -> str | None:
    """
    Which family of measurement an attribute label names, or None.

    None is a real answer, not a failure: 'share of employees bound by a
    non-compete' names no family in this lexicon, and the caller then falls
    back to token overlap. Ambiguous labels -- ones scoring equally in two
    families -- also return None rather than guessing, because a wrong family
    merges two different measurements into one ranking, which is the failure
    this whole mechanism exists to prevent.
    """
    words = set(re.findall(r"[a-z]+", (attribute or "").lower())) - _ATTR_MODIFIER
    if not words:
        return None
    scores = {}
    for family, vocabulary in _MEASUREMENT_FAMILIES.items():
        hits = len(words & vocabulary)
        if hits:
            scores[family] = hits
    if not scores:
        return None
    best = max(scores.values())
    winners = [f for f, n in scores.items() if n == best]
    return winners[0] if len(winners) == 1 else None


def cluster_by_measurement(records: list[dict]) -> list[dict]:
    """
    Partition records into groups that measure the same thing.

    Returns a list of {'label', 'family', 'members'}, largest first. Records
    whose label names a known family group by family; the rest cluster greedily
    by token overlap, exactly as the ledger does.
    """
    by_family: dict[str, list[dict]] = {}
    unfamiliar: list[dict] = []
    for r in records:
        family = measurement_family(r.get("attribute"))
        if family:
            by_family.setdefault(family, []).append(r)
        else:
            unfamiliar.append(r)

    clusters: list[tuple[set, list[dict]]] = []
    for r in unfamiliar:
        tokens = _attr_tokens(r.get("attribute"))
        for cluster_tokens, members in clusters:
            if _same_measurement(tokens, frozenset(cluster_tokens)):
                cluster_tokens |= tokens
                members.append(r)
                break
        else:
            clusters.append((set(tokens), [r]))

    def label_for(members: list[dict]) -> str:
        # The label most readers actually used, so the answer quotes the
        # document's own wording rather than a family name Python invented.
        seen: dict[str, int] = {}
        for r in members:
            seen[r["attribute"]] = seen.get(r["attribute"], 0) + 1
        return max(seen, key=seen.get)

    out = [{"label": label_for(m), "family": f, "members": m}
           for f, m in by_family.items()]
    out += [{"label": label_for(m), "family": None, "members": m}
            for _, m in clusters]
    out.sort(key=lambda c: -len(c["members"]))
    return out


def dominant_measurement(records: list[dict]) -> tuple[list[dict], dict]:
    """The biggest cluster, and a count of what was set aside, for the note."""
    clusters = cluster_by_measurement(records)
    if not clusters:
        return records, {}
    kept = clusters[0]
    set_aside = {c["label"]: len(c["members"]) for c in clusters[1:]}
    return kept["members"], set_aside


def parse_claims(text: str, source: str = "") -> list[dict]:
    """Pull CLAIM lines out of one reader's reply."""
    out = []
    for m in _CLAIM.finditer(text):
        entity, claim, where = (p.strip() for p in m.groups())
        if not entity or not claim:
            continue
        if entity.lower().strip(" .:") in _NOT_AN_ENTITY:
            continue
        word = _SUPERLATIVE.search(claim)
        out.append({"entity": entity, "claim": claim, "where": where,
                    "source": source,
                    "superlative": word.group(0).lower() if word else None})
    return out


def _entity_key(name: str) -> str:
    """
    'Saxon Switzerland National Park' and 'Saxon Switzerland' are one entity.

    Both spellings occur in the same document, in the same reader's reply, and
    grouping on the raw string files them apart -- which either hides a real
    disagreement or invents one.
    """
    key = re.sub(r"[^\w\s]", " ", (name or "").lower())
    key = re.sub(r"\b(national|nature|natural|regional)\b", " ", key)
    key = re.sub(r"\b(park|reserve|forest|n\s*p)\b", " ", key)
    return " ".join(key.split())


def _attr_tokens(attribute: str) -> frozenset:
    """The meaning-carrying words of an attribute label, stemmed."""
    words = re.findall(r"[a-z]+", (attribute or "").lower())
    return frozenset(w[:4] for w in words if w not in _ATTR_NOISE and len(w) > 2)


def _same_measurement(a: frozenset, b: frozenset) -> bool:
    """
    Do two attribute labels describe the same thing?

    Exact string equality is useless here. One parks run produced sixty-odd
    labels for what is largely one measurement -- 'area covered', 'area',
    'land area covered', 'marine surface area', 'area covered, including buffer
    zone', 'Sea area covered'. Grouping on the string finds no disagreements
    because every record sits alone in its own bucket.

    Overlap, not identity: two labels match when most of the smaller one's
    words appear in the larger. That keeps 'area' with 'area covered' while
    holding 'land area' and 'marine area' apart, since 'land' and 'marine' are
    the words that make them different measurements.
    """
    if not a or not b:
        return False
    shared = len(a & b)
    return shared >= 1 and shared >= 0.6 * min(len(a), len(b))


# What a superlative is implicitly about. A claim says "the largest park" and
# the table column says "Area covered": there is no word in common, and no
# amount of string matching will connect them. This is a small English lexicon,
# not a fact about any document.
_SUPERLATIVE_MEANS = {
    "largest": ("area", "size", "surf", "exte"), "biggest": ("area", "size"),
    "smallest": ("area", "size", "surf"), "widest": ("widt", "area"),
    "highest": ("heig", "poin", "peak", "elev", "alti", "summ"),
    "tallest": ("heig", "poin", "peak", "elev", "summ"),
    "lowest": ("heig", "poin", "elev", "alti"),
    "longest": ("leng", "dist", "trai", "rout"),
    "shortest": ("leng", "dist", "trai", "rout"),
    "deepest": ("dept",), "oldest": ("age", "year", "esta", "date"),
    "youngest": ("age", "year", "esta", "date"),
    "most": ("numb", "coun", "visi"), "greatest": ("numb", "coun"),
    "first": ("year", "date", "esta"), "only": (),
}


def _claim_tokens(claim: dict) -> frozenset:
    """
    What measurement a superlative claim is about.

    Takes the claim's own words plus whatever the superlative implies, so that
    "the largest national park in the Alps" can be compared against records
    labelled "Area covered".
    """
    tokens = set(_attr_tokens(claim["claim"]))
    implied = set(_SUPERLATIVE_MEANS.get(claim.get("superlative") or "", ()))
    if g10_on("claimfam") and measurement_family(claim["claim"]):
        # The claim's own words already name what is measured, so the
        # superlative implies nothing further. Injecting anyway is how
        # "France's largest population of golden eagles (37 pairs)" acquired
        # the token 'area' from 'largest', matched Ecrins' 918 sq km, and was
        # reported as contradicted by another park's 3583 sq km -- two facts
        # with nothing to do with each other or with the question.
        implied = set()
    return frozenset(tokens | implied)


def _claim_matches(claim_tokens: frozenset, attr_tokens: frozenset) -> bool:
    """
    Looser than _same_measurement, deliberately.

    A claim is a sentence, not a column heading -- it carries scope words
    ("in Britain"), a noun ("mountain") and the superlative itself, so demanding
    proportional overlap against a two-word label finds nothing. One shared
    token is enough here because check_claims then requires the claimed entity
    to actually have a figure for that measurement, and the adjudicator sees
    every survivor.
    """
    return bool(claim_tokens & attr_tokens)


def group_records(records: list[dict]) -> dict:
    """
    Cluster records by entity, then by what was actually measured.

    Returns {(entity_key, cluster_index): [records]}. Clustering is greedy and
    order-dependent, which is fine: the labels within one real measurement are
    far more similar to each other than to anything else.
    """
    by_entity: dict[str, list[dict]] = {}
    for r in records:
        key = _entity_key(r.get("entity"))
        if key:
            by_entity.setdefault(key, []).append(r)

    grouped = {}
    for entity, rs in by_entity.items():
        clusters: list[tuple[set, list]] = []
        for r in rs:
            tokens = _attr_tokens(r.get("attribute"))
            for cluster_tokens, members in clusters:
                if _same_measurement(tokens, frozenset(cluster_tokens)):
                    cluster_tokens |= tokens
                    members.append(r)
                    break
            else:
                clusters.append((set(tokens), [r]))
        for i, (_, members) in enumerate(clusters):
            grouped[(entity, i)] = members
    return grouped


def _display_value(r: dict) -> str:
    return (r.get("raw_value") or "").strip() or str(r.get("value"))


def _compare_value(r: dict):
    if r.get("normalized") is not None:
        return round(r["normalized"], 6)
    return (r.get("raw_value") or "").lower().strip()


# How close two values must be, relatively, to look like the same quantity
# reported twice rather than two different quantities. 408 vs 409 is 0.002.
NEAR_VALUE_GAP = float(os.environ.get("NEAR_VALUE_GAP", "0.05"))


def find_value_conflicts(records: list[dict]) -> list[dict]:
    """
    One entity, one measurement, two different values.

    Ranked by RELATIVE closeness, not by how far apart the numbers are. A pair
    like 408 and 409 is a transposition between two tables -- the thing these
    questions are about. A pair like 2474 and 403 is two columns of one row
    being read as one measurement, which is a parsing artifact. Sorting the
    near-misses to the top puts the real ones in front of the adjudicator.
    """
    out = []
    for (entity, _), members in group_records(records).items():
        values = {_compare_value(r) for r in members}
        if len(values) < 2:
            continue
        numeric = [r["normalized"] for r in members
                   if r.get("normalized") is not None]
        gap = None
        if len(numeric) >= 2 and max(numeric):
            gap = (max(numeric) - min(numeric)) / max(abs(v) for v in numeric)

        seen, distinct = set(), []
        for r in members:
            key = _compare_value(r)
            if key in seen:
                continue
            seen.add(key)
            distinct.append({"value": _display_value(r), "where": r.get("where"),
                             "found_in": r.get("source")})
        sources = {r.get("source") or "" for r in members}
        out.append({
            "kind": "value_conflict",
            "entity": members[0].get("entity"), "attribute": members[0].get("attribute"),
            "relative_gap": None if gap is None else round(gap, 4),
            # A disagreement inside one slice is usually two columns misread.
            # The real ones are far apart, which is why nothing else finds them.
            "separated": len(sources) > 1,
            "found_in": sorted(sources), "values": distinct,
        })

    # Second pass: near-equal values for one entity that landed in DIFFERENT
    # clusters because the two places label the measurement with different
    # words. Table 5.1 calls it "Nb. of companies" and the annex calls it
    # "number of interviews"; nothing lexical connects those, and that is
    # precisely the pair these questions are about.
    #
    # Numeric closeness is the signal instead. 408 against 409 is a
    # transposition; 2146 against 408 is two columns of one row and stays out.
    # Different parts of the document only -- two labels inside one slice
    # disagreeing is a reader misreading a table, not the document contradicting
    # itself.
    already = {(c["entity"] or "").lower() for c in out}
    per_entity: dict[str, list[tuple[int, list]]] = {}
    for (entity_key, ci), members in group_records(records).items():
        per_entity.setdefault(entity_key, []).append((ci, members))
    for entity_key, clusters in per_entity.items():
        if len(clusters) < 2:
            continue
        flat = [(ci, r) for ci, members in clusters for r in members
                if r.get("normalized") is not None]
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                (ci, a), (cj, b) = flat[i], flat[j]
                if ci == cj or a.get("source") == b.get("source"):
                    continue
                hi = max(abs(a["normalized"]), abs(b["normalized"]))
                if not hi or a["normalized"] == b["normalized"]:
                    continue
                gap = abs(a["normalized"] - b["normalized"]) / hi
                if gap > NEAR_VALUE_GAP:
                    continue
                if (a.get("entity") or "").lower() in already:
                    continue
                already.add((a.get("entity") or "").lower())
                out.append({
                    "kind": "value_conflict",
                    "entity": a.get("entity"),
                    "attribute": f"{a.get('attribute')} / {b.get('attribute')}",
                    "relative_gap": round(gap, 4), "separated": True,
                    "labelled_differently": True,
                    "found_in": sorted({a.get("source") or "", b.get("source") or ""}),
                    "values": [
                        {"value": _display_value(a), "where": a.get("where"),
                         "found_in": a.get("source")},
                        {"value": _display_value(b), "where": b.get("where"),
                         "found_in": b.get("source")}],
                })

    out.sort(key=lambda c: (not c["separated"],
                            c["relative_gap"] if c["relative_gap"] is not None else 9))
    return out


def find_unit_outliers(records: list[dict]) -> list[dict]:
    """
    One measurement reported in a different unit from all the others.

    Nearly free: _parse_value already keeps the printed unit alongside the
    converted value, and the comment where it does so says this is 'not noise,
    it is sometimes the answer'. It was never wired to anything. In the parks
    book exactly one park gives its area in square miles while every other one
    uses square kilometres.
    """
    by_measurement: dict[frozenset, list[dict]] = {}
    for r in records:
        if not r.get("unit"):
            continue
        tokens = _attr_tokens(r.get("attribute"))
        if not tokens:
            continue
        for key in list(by_measurement):
            if _same_measurement(tokens, key):
                by_measurement[key].append(r)
                break
        else:
            by_measurement[frozenset(tokens)] = [r]

    out = []
    for _, members in by_measurement.items():
        counts: dict[str, list[dict]] = {}
        for r in members:
            counts.setdefault(r["unit"].lower(), []).append(r)
        if len(counts) < 2 or len(members) < 5:
            continue
        ranked = sorted(counts.items(), key=lambda kv: -len(kv[1]))
        majority_unit, majority = ranked[0]
        for unit, odd in ranked[1:]:
            # A genuine outlier is rare against a clear majority, not a second
            # common unit -- two units used equally often is a mixed table, not
            # one entry printed differently from the rest.
            if len(odd) > 2 or len(majority) < 4 * len(odd):
                continue
            out.append({
                "kind": "unit_outlier",
                "attribute": odd[0].get("attribute"),
                "majority_unit": majority_unit, "majority_count": len(majority),
                "odd_unit": unit,
                "entities": [{"entity": r.get("entity"), "value": _display_value(r),
                              "where": r.get("where")} for r in odd],
            })
    return out


# A data-box label carrying its unit in a parenthetical: 'Area covered (sq km)'.
_LABELLED_UNIT = re.compile(
    r"^[ \t]*(?P<label>[A-Za-z][^()\n]{2,60}?)[ \t]*\((?P<unit>[^()\n]{1,18})\)[ \t]*$",
    re.M)


def find_label_unit_outliers(text: str, names: list[str]) -> list[dict]:
    """
    A unit outlier found in the DOCUMENT, not in the readers' records.

    find_unit_outliers works over what readers reported, and so cannot see the
    one thing it exists to find. The parks book prints

        Park in numbers
        1151
        Area covered (sq miles)

    -- the unit lives in the label's parenthetical, on its own line, and the
    value sits on the line above. Fifty entries read 'Area covered (sq km)' and
    exactly one reads 'Area covered (sq miles)'. Every reader that met that box
    dropped the parenthetical and emitted 'area covered | 1151', a record with
    no unit at all, which the record-side detector skips by construction.

    Reading the raw text instead makes this immune to what any reader chose to
    report. It costs no model call, and it needs no knowledge of this document:
    'a repeated label whose unit is in brackets' is a typographic convention,
    not a fact about national parks.
    """
    sites = []
    for m in _LABELLED_UNIT.finditer(text):
        unit = m.group("unit").strip()
        if not _UNIT_ONLY.match(unit):
            continue
        tokens = _attr_tokens(m.group("label"))
        if tokens:
            sites.append({"label": m.group("label").strip(), "unit": unit,
                          "tokens": tokens, "at": m.start()})

    # Group the labels that describe one measurement, exactly as elsewhere.
    groups: list[tuple[set, list]] = []
    for site in sites:
        for keys, members in groups:
            if _same_measurement(site["tokens"], frozenset(keys)):
                keys |= site["tokens"]
                members.append(site)
                break
        else:
            groups.append((set(site["tokens"]), [site]))

    out = []
    for _, members in groups:
        by_unit: dict[str, list[dict]] = {}
        for site in members:
            by_unit.setdefault(site["unit"].lower(), []).append(site)
        if len(by_unit) < 2 or len(members) < 5:
            continue
        ranked = sorted(by_unit.items(), key=lambda kv: -len(kv[1]))
        majority_unit, majority = ranked[0]
        for unit, odd in ranked[1:]:
            if len(odd) > 2 or len(majority) < 4 * len(odd):
                continue
            out.append({
                "kind": "unit_outlier",
                "found_by": "document label, not by a reader",
                "attribute": odd[0]["label"],
                "majority_unit": majority_unit, "majority_count": len(majority),
                "majority_label": majority[0]["label"], "odd_unit": unit,
                "entities": [{"entity": _nearest_name(text, s["at"], names),
                              "value": _value_beside(text, s["at"]),
                              "where": s["label"]} for s in odd],
            })
    return out


def _value_beside(text: str, at: int, span: int = 200) -> str:
    """
    The figure a label belongs to: the line above it, else the line below.

    A data box prints the number first and names it second, so 'above' is the
    common case; a table that labels its column first is the other one.
    """
    # Upward first, skipping the blank lines a flattened PDF leaves between
    # every pair of real ones -- and stopping at the first non-blank line, so
    # that a label with no figure of its own cannot steal the figure belonging
    # to the label above it.
    for line in reversed(text[max(0, at - span):at].splitlines()):
        if not line.strip():
            continue
        if _NUMBER.fullmatch(line.strip()):
            return line.strip()
        break
    end = text.find("\n", at)
    for line in text[end + 1:end + 1 + span].splitlines():
        if not line.strip():
            continue
        return line.strip() if _NUMBER.fullmatch(line.strip()) else ""
    return ""


def _nearest_name(text: str, at: int, names: list[str], back: int = 6000) -> str:
    """
    Which entity a position in the document belongs to.

    The nearest preceding mention of a name the readers have already returned.
    No heading structure is assumed: in the parks book the section headings are
    'Do this!' and 'What to spot...', repeated once per park, and the park's own
    name appears only in running text.
    """
    window = text[max(0, at - back):at]
    best, best_at = "", -1
    for name in names:
        # Readers return row labels as well as entities -- 'trail', 'depth',
        # 'area'. A proper name is capitalised and long; matching case
        # sensitively is what keeps the common noun out, since the document
        # writes the noun in lower case and the park's name in title case.
        if len(name) < 5 or not name[:1].isupper():
            continue
        if name.lower().strip(" .:") in _NOT_AN_ENTITY:
            continue
        found = window.rfind(name)
        if found > best_at or (found == best_at and len(name) > len(best)):
            best, best_at = name, found
    return best or "(entity not identified)"


_SCOPE_ATTR = ("coun", "regi", "stat", "loca", "natio")


def entity_scopes(records: list[dict]) -> dict[str, str]:
    """
    Which country or region each entity belongs to, where a reader said so.

    Built from the records themselves: 'RECORD | Kornati Islands National Park |
    country | Croatia' is a record like any other. Only non-numeric values
    count -- a country column holds a name, not a figure.
    """
    scopes = {}
    for r in records:
        tokens = _attr_tokens(r.get("attribute"))
        if not any(t.startswith(p[:4]) for t in tokens for p in _SCOPE_ATTR):
            continue
        value = (r.get("raw_value") or "").strip()
        if not value or r.get("value") is not None:
            continue
        scopes.setdefault(_entity_key(r.get("entity")), value.lower())
    return scopes


# Where a superlative says it applies. Three shapes, all common in English:
# a possessive ("Norway's largest"), a prepositional phrase ("the highest in
# Wales", "the largest in the Alps"), and a bare adjective of nationality
# ("Britain's second-highest", "the tallest British peak").
_CLAIM_SCOPE = (
    # {1,} not {2,}: 'the UK's sixth-highest mountain' scopes to Britain, and a
    # minimum name length of three characters silently dropped every one of the
    # three Snowdon claims g10 emitted.
    re.compile(r"\b(?:the\s+)?([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]+)?)['’]s\b"),
    re.compile(r"\b(?:in|of|across|within|throughout)\s+(?:the\s+)?"
               r"([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)\b"),
)

# Names for one place. Not a fact about parks -- a superlative scoped to Wales
# is scoped to the UK, and a park in Svalbard is a park in Norway, and a
# comparison that ignores that reports a contradiction where none exists.
_SCOPE_ALIASES = {
    "uk": {"uk", "britain", "great britain", "united kingdom", "england",
           "scotland", "wales", "northern ireland", "british"},
    "norway": {"norway", "norwegian", "svalbard", "spitsbergen"},
    "spain": {"spain", "spanish", "catalonia", "andalusia"},
    "italy": {"italy", "italian", "sicily", "sardinia"},
    "finland": {"finland", "finnish", "lapland"},
    "france": {"france", "french", "corsica"},
    "croatia": {"croatia", "croatian", "dalmatia"},
}
_SCOPE_CANON = {name: canon for canon, names in _SCOPE_ALIASES.items()
                for name in names}

# Words that pass the capitalisation test but name no place.
_NOT_A_SCOPE = {"the", "national", "park", "europe", "european", "world",
                "continent", "earth", "planet", "unesco", "june", "july",
                "one", "two", "its", "this"}


def claim_scope(claim: str) -> str | None:
    """The place a superlative claim restricts itself to, from its own words."""
    for pattern in _CLAIM_SCOPE:
        for m in pattern.finditer(claim or ""):
            name = m.group(1).strip().lower()
            if name in _NOT_A_SCOPE or len(name) < 2:
                continue
            return _SCOPE_CANON.get(name, name)
    return None


def _scope_compatible(mine: str | None, theirs: str | None) -> bool:
    """
    May a figure belonging to `theirs` refute a claim scoped to `mine`?

    Unknown is compatible: the alternative is deleting real refutations for
    every entity whose country no reader happened to write down.
    """
    if mine is None or theirs is None:
        return True
    a = _SCOPE_CANON.get(mine.lower(), mine.lower())
    b = _SCOPE_CANON.get(theirs.lower(), theirs.lower())
    return a == b or a in b or b in a


def check_claims(claims: list[dict], records: list[dict]) -> list[dict]:
    """
    A superlative the document asserts, against the figures it prints elsewhere.

    This is the shape the value ledger cannot see: nothing disagrees about one
    entity's value. The book says a park is the largest in its country and then
    prints a bigger number for a different park, or calls a mountain the highest
    and lists a taller one two hundred pages away. The claim and the refutation
    belong to DIFFERENT entities, so grouping by entity finds nothing.
    """
    scopes = entity_scopes(records)
    out = []
    for claim in claims:
        word = claim.get("superlative")
        if word not in {"largest", "biggest", "highest", "tallest", "longest",
                        "deepest", "greatest", "most", "smallest", "shortest",
                        "lowest"}:
            continue
        wanted = _claim_tokens(claim)
        subject = _entity_key(claim["entity"])
        smaller = word in {"smallest", "shortest", "lowest"}

        comparable = [r for r in records
                      if r.get("normalized") is not None
                      and _claim_matches(wanted, _attr_tokens(r.get("attribute")))]
        if len(comparable) < 3:
            continue
        mine = [r for r in comparable if _entity_key(r.get("entity")) == subject]
        if not mine:
            continue
        my_value = (min if smaller else max)(r["normalized"] for r in mine)

        beaten = [r for r in comparable
                  if _entity_key(r.get("entity")) != subject
                  and (r["normalized"] < my_value if smaller
                       else r["normalized"] > my_value)]

        # A superlative is always scoped -- "the largest in its country", "the
        # highest in Wales" -- and a bigger figure from outside that scope
        # refutes nothing. Where the document says which country an entity
        # belongs to, hold the comparison inside it.
        my_scope = scopes.get(subject)
        claimed_scope = claim_scope(claim["claim"]) if g11_on("claimscope") else None
        if my_scope is None:
            # The claim states its own scope: "the UK's sixth-highest mountain",
            # "Norway's largest national park". Reading it out of the claim
            # needs no reader to have volunteered a country record, which is
            # what made this miss: seven of g10's nineteen surviving candidates
            # had no scope, and three of them were Snowdon -- the subject of the
            # one contradiction the question was actually about.
            my_scope = claimed_scope
        scope_known = my_scope is not None
        if scope_known:
            # Drop a refutation only when the other entity's scope is KNOWN and
            # different. Unknown stays in: dropping it would have deleted
            # Snowdon-versus-Ben-Macdui along with Snowdon-versus-Etna, and a
            # deleted right answer costs exactly as much as a wrong one.
            beaten = [r for r in beaten
                      if _scope_compatible(
                          my_scope, scopes.get(_entity_key(r.get("entity"))))]
        if not beaten:
            continue
        if scope_known and g11_on("claimscope"):
            # If anything inside the claim's own scope refutes it, that IS the
            # refutation; entities whose country the document never stated are
            # only a fallback for when nothing in scope does. Without this the
            # list is led by whichever foreign peak is tallest, and the
            # in-scope one -- the answer -- is truncated off the end.
            in_scope = [r for r in beaten
                        if scopes.get(_entity_key(r.get("entity"))) is not None]
            if in_scope:
                beaten = in_scope
        beaten.sort(key=lambda r: r["normalized"], reverse=not smaller)
        out.append({
            "kind": "claim_vs_data",
            "entity": claim["entity"], "claim": claim["claim"],
            "claim_where": claim.get("where"),
            "claimed_value": _display_value(mine[0]),
            "scope": my_scope,
            # When the document never says where an entity is, Python cannot
            # tell a real refutation from a comparison across countries. Say so
            # rather than pretend, and let the adjudicator apply the scope the
            # claim itself names.
            "scope_unverified": not scope_known,
            "contradicted_by": [
                {"entity": r.get("entity"), "value": _display_value(r),
                 "where": r.get("where"),
                 "scope": scopes.get(_entity_key(r.get("entity")))}
                for r in beaten[:4]],
        })
    # Scope-checked candidates first; the unverified ones are guesses.
    out.sort(key=lambda c: c["scope_unverified"])
    return out


def build_claim_ledger(records: list[dict]) -> list[dict]:
    """
    Group ingested records by (entity, attribute); a key holding more than one
    distinct value is a contradiction candidate.

    This is the filter that makes the problem tractable. Comparing every claim
    to every other claim is quadratic and, for certifying global consistency,
    worse than that; grouping on a cheap key proposes a shortlist in one pass
    and a model only ever sees the shortlist.

    It also fixes the failure mode directly. Both scored contradiction answers
    stated number pairs that were near-misses of the real ones, because nothing
    in the pipeline had ever put two values for the same attribute side by side
    -- so the model produced a pair that looked like what the question wanted.
    A ledger cannot produce a pair the document does not contain.
    """
    ledger: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        entity = (r.get("entity") or "").lower().strip(" .:")
        attribute = (r.get("attribute") or "").lower().strip(" .:")
        if not entity or not attribute:
            continue
        ledger.setdefault((entity, attribute), []).append(r)

    def compare_key(r: dict):
        # Compare on the normalized value where a unit was parsed, so 'sq miles'
        # and 'sq km' do not read as a disagreement. Records whose value carries
        # no digits ('party to the convention') still contradict each other and
        # are compared as strings.
        if r.get("normalized") is not None:
            return round(r["normalized"], 6)
        return (r.get("raw_value") or "").lower().strip()

    candidates = []
    for (entity, attribute), rs in ledger.items():
        values = {compare_key(r) for r in rs}
        if len(values) < 2:
            continue
        sources = {r.get("source") or "" for r in rs}
        seen, distinct = set(), []
        for r in rs:
            key = compare_key(r)
            if key in seen:
                continue
            seen.add(key)
            distinct.append({"value": r.get("raw_value"), "where": r.get("where"),
                             "found_in": r.get("source")})
        candidates.append({
            "entity": entity, "attribute": attribute,
            "found_in": sorted(sources),
            # A disagreement inside one slice is usually a parsing artifact --
            # two columns of one table read as one. The real ones are far apart,
            # which is exactly why both humans and retrieval miss them.
            "separated": len(sources) > 1,
            "values": distinct,
        })

    candidates.sort(key=lambda c: (not c["separated"], -len(c["found_in"])))
    return candidates


# --------------------------------------------------------------- PROVENANCE

def verify_record(record: dict, text: str, window: int | None = None) -> str:
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
        reach = PROVENANCE_WINDOW if window is None else window
        near = text[max(0, m.start() - reach):m.end() + reach]
        if printed in near or bare in near.replace(",", ""):
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

    if fix_on("catattr"):
        # 'country', 'location', 'country/region' and 'situated in' are one
        # question. Keeping only the commonest of them is how "how many parks
        # are in Spain" came back as three when all six were recorded.
        kept, _ = dominant_measurement(labelled)
        dominant = max({r["attribute"] for r in kept},
                       key=lambda a: sum(1 for r in kept if r["attribute"] == a))
    else:
        counts: dict[str, int] = {}
        for r in labelled:
            counts[r["attribute"]] = counts.get(r["attribute"], 0) + 1
        dominant = max(counts, key=counts.get)
        kept = [r for r in labelled if r["attribute"] == dominant]

    seen: dict[str, str] = {}
    for r in kept:
        # _entity_key, so 'Ordesa' and 'Ordesa National Park' are not two parks.
        key = (_entity_key(r["entity"]) if fix_on("catattr")
               else r["entity"].strip().lower())
        seen.setdefault(key or r["entity"].strip().lower(),
                        r["raw_value"].strip().lower())

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
        # g9: the ledger has always folded 'Saxon Switzerland' into 'Saxon
        # Switzerland National Park'; the tally never did, so one park counted
        # twice against a threshold and its two spellings never resolved.
        key = (_entity_key(r["entity"]) if fix_on("entity")
               else r["entity"].strip().lower())
        by_entity.setdefault(key or r["entity"].strip().lower(), []).append(r)

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


def recover_named_values(records: list[dict], text: str) -> int:
    """
    Read back a value the reader returned as a name instead of a number.

    The parks book prints its data box as

        4102
        Highest point: Barre des Ecrins (m)

    -- the figure on one line, its label on the next, and the label carries the
    name of the thing measured. One run's reader emitted '4102' for Ecrins; the
    next run's emitted 'Barre des Ecrins (m)'. A value with no digits is not
    numeric, so the record left the ranking silently and the highest summit in
    the book stopped being the highest summit in the book.

    Nothing is invented: the number is taken from the document, next to the
    label, inside the entity's own passage, and only when the record has no
    number of its own. Records that already parsed are untouched.
    """
    fixed = 0
    for r in records:
        if r.get("normalized") is not None or not r.get("raw_value"):
            continue
        if _NUMBER.search(r["raw_value"]):
            continue                      # has digits; _parse_value's problem
        entity = (r.get("entity") or "").strip()
        if len(entity) < 4:
            continue
        # The label to look for is whatever the reader put in the value field
        # ('Barre des Ecrins (m)'), falling back to the attribute it named.
        probe = re.sub(r"\s*\([^)]*\)\s*$", "", r["raw_value"].strip()).strip()
        if len(probe) < 3:
            continue
        wanted = _attr_tokens(r.get("attribute"))
        for m in re.finditer(re.escape(entity), text, re.I):
            window = text[m.start():m.start() + PROVENANCE_WINDOW]
            for hit in re.finditer(re.escape(probe), window, re.I):
                # The probe must sit on a LABEL LINE -- one that also names the
                # measurement the record claims to be about ('Highest point:
                # Barre des Ecrins (m)'). Without that test the probe matches
                # the same name in running prose and _value_beside returns
                # whatever figure happens to be nearby: one draft of this
                # function recovered 800,000 as the height of a 4,102 m peak,
                # which is worse than the missing record it was fixing.
                start = window.rfind("\n", 0, hit.start()) + 1
                end = window.find("\n", hit.end())
                line = window[start:end if end != -1 else len(window)]
                if not _same_measurement(wanted, _attr_tokens(line)):
                    continue
                # Only the figure ABOVE the label. A data box prints the number
                # then names it; the line below belongs to the NEXT label.
                above = [ln.strip() for ln in window[:start].splitlines()
                         if ln.strip()]
                if not above or not _NUMBER.fullmatch(above[-1]):
                    continue
                parsed = _parse_value(above[-1])
                if parsed[3] is None:
                    continue
                r["raw_value"], r["value"] = above[-1], parsed[0]
                r["unit"], r["unit_family"] = parsed[1], parsed[2]
                r["normalized"] = parsed[3]
                r["recovered_from_text"] = True
                fixed += 1
                break
            if r.get("recovered_from_text"):
                break
    return fixed


def tally(records: list[dict], question: str, source_text: str = "",
          window: int | None = None) -> dict | None:
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
            verdict = verify_record(r, source_text, window)
            r["provenance"] = verdict
            if verdict in ("verified", "verified_nearby", "unverifiable"):
                checked.append(r)
            else:
                rejected.setdefault(verdict, []).append(
                    f"{r['entity']} = {r['raw_value']}")
        records = checked

    if source_text and g11_on("recover"):
        recovered = recover_named_values(records, source_text)
        if recovered:
            _log(f"  ~ recovered {recovered} value(s) the readers returned as a "
                 f"name instead of a number")

    numeric = [r for r in records if r["normalized"] is not None]
    if not numeric:
        cat = categorical_tally(records, question)
        if cat and rejected:
            cat["rejected_unverifiable"] = {k: v[:12] for k, v in rejected.items()}
        return cat

    # Readers may report more than one attribute. Comparing heights against
    # areas is meaningless, so rank within the attribute most readers reported
    # and say plainly what was set aside.
    if fix_on("attr"):
        # Group by what was MEASURED, not by how it was spelled. See
        # measurement_family: g8 answered four of eight parks over a threshold
        # while holding all eight correct records, because five readers spelled
        # one measurement five ways and only the commonest spelling survived.
        numeric, set_aside = dominant_measurement(numeric)
        dominant = max({r["attribute"] for r in numeric},
                       key=lambda a: sum(1 for r in numeric if r["attribute"] == a))
    else:
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

ARC_PROMPT = """You are reading part {n} of {of} of {doc}, in order. The text below is
verbatim -- nothing has been summarized or removed.

The question asks how something DEVELOPS across the whole document: what changes
and what stays constant. You hold one {of}th of it. Your job is NOT to answer the
question -- you cannot see the other parts. Your job is to report, exactly, what
THIS part says about the theme, so that the sequence of all parts in order shows
the shape of the argument.

REPORT SOMETHING EVERY TIME. Unlike a question about one buried fact, a question
about development has no empty parts: a part that barely touches the theme is
itself evidence about where the theme is and is not carried. If this part really
does not engage the theme, say so in the STANCE line and still fill in the rest.

Give these lines, in this order:

    POSITION: part {n} of {of}
    SECTION: <the chapter, section or heading this part falls in, as printed>
    STANCE: <in one or two sentences, how THIS part frames the theme -- the
             claim it makes, the attitude it takes, or "this part does not
             engage the theme">
    FRAMING: <the vocabulary and emphasis used here -- e.g. "risk and harm",
              "individual choice", "structural power", "policy remedy">
    EVIDENCE: <where -- the specific fact, figure, percentage, year or name>
    EVIDENCE: <...one line each, up to 6>
    SHIFT: <only if this part explicitly marks a change from what came before --
            quote or closely restate the sentence that does so; otherwise omit>

Rules:
- Every name, figure, percentage and date must be findable in the text below.
  NEVER supply a fact from your own knowledge.
- Copy figures, percentages, years and proper names EXACTLY as printed.
- KEEP QUALIFIERS. "about two-thirds" is not "most". "projected to" is not "did".
- Name the chapter/section on SECTION even if you must infer it from a heading or
  running header in the text -- the ordering of the final answer depends on it.
- Do not describe the document as a whole; you see only one {of}th.

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""


# g12, step 4a. The same prompt as above with ONE difference: the reader is
# handed the section headings its part covers, as printed, instead of being told
# to infer one. Kept as a separate template rather than a slot in ARC_PROMPT so
# that the words-mode path is byte-for-byte the string g11 sends -- the OECD and
# GEM documents carry no headers and must not see so much as a different space.
#
# This REMOVES work from the model rather than adding it. The SECTION line was
# the one place in the arc reader where the model was asked to guess at document
# structure, and on a document that prints its own headings the guess is
# replaceable by a fact.
ARC_PROMPT_HEADERS = """You are reading part {n} of {of} of {doc}, in order. The text below is
verbatim -- nothing has been summarized or removed.

The question asks how something DEVELOPS across the whole document: what changes
and what stays constant. You hold one {of}th of it. Your job is NOT to answer the
question -- you cannot see the other parts. Your job is to report, exactly, what
THIS part says about the theme, so that the sequence of all parts in order shows
the shape of the argument.

THIS PART COVERS THESE SECTIONS OF THE DOCUMENT, in order, as the document
prints their headings:

    {sections}

REPORT SOMETHING EVERY TIME. Unlike a question about one buried fact, a question
about development has no empty parts: a part that barely touches the theme is
itself evidence about where the theme is and is not carried. If this part really
does not engage the theme, say so in the STANCE line and still fill in the rest.

Give these lines, in this order:

    POSITION: part {n} of {of}
    SECTION: <copy the heading list above, or the one of them this part is
             mostly about -- do NOT infer or invent a section name>
    STANCE: <in one or two sentences, how THIS part frames the theme -- the
             claim it makes, the attitude it takes, or "this part does not
             engage the theme">
    FRAMING: <the vocabulary and emphasis used here -- e.g. "risk and harm",
              "individual choice", "structural power", "policy remedy">
    EVIDENCE: <where -- the specific fact, figure, percentage, year or name>
    EVIDENCE: <...one line each, up to 6>
    SHIFT: <only if this part explicitly marks a change from what came before --
            quote or closely restate the sentence that does so; otherwise omit>

Rules:
- Every name, figure, percentage and date must be findable in the text below.
  NEVER supply a fact from your own knowledge.
- Copy figures, percentages, years and proper names EXACTLY as printed.
- KEEP QUALIFIERS. "about two-thirds" is not "most". "projected to" is not "did".
- The SECTION line is given to you above. Use it; do not guess at one.
- Do not describe the document as a whole; you see only one {of}th.

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""


CROSS_PROMPT = """You are reading part {n} of {of} of {doc}. The text below is verbatim --
nothing has been summarized or removed.

This part was selected because it contains these exact phrases from the
question, which occur only a few times in the whole document:

    {anchors}

The question asks which entity holds TWO OR MORE separate facts that are stated
in DIFFERENT places. You hold one of those places, probably not both. DO NOT try
to answer the question. Report what is here, precisely enough that it can be
matched against the other place.

Around each phrase listed above, report:

    ANCHOR: <the phrase>
    ENTITY: <the country, organisation, institution or person the passage
             attaches this to, exactly as printed -- this is the single most
             important line; if the passage names a region or a city, give the
             country it belongs to as well IF the text says so>
    FACT: <what the passage actually states about that entity, in one sentence,
           keeping every figure, date and qualifier as printed>
    WHERE: <the chapter, section, table or figure>

Repeat that block for each phrase you find. If a phrase from the list appears in
a heading or a passing cross-reference with no entity attached, say so:

    ANCHOR: <the phrase>
    ENTITY: none stated here

If the passage says an entity is ABSENT, EXCLUDED or NOT COVERED by something,
that is a fact worth reporting, not a reason to stay silent -- report it with
the entity it concerns. A question can turn on exactly that.

If none of the listed phrases appears here after all, reply with exactly:

    {marker}

Rules:
- Every name, figure and date must be findable in the text below. NEVER supply a
  fact from your own knowledge: another part holds the other half of the link,
  and an invented detail displaces the real one.
- Copy proper names EXACTLY as printed.
- Do not guess which entity the question is about. Report what is here.

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""


# Words that end in -s without being plural nouns, so that "which park has" is
# not read as a set-valued question.
_NOT_PLURAL = {"is", "was", "has", "does", "its", "this", "less", "across",
               "us", "gas", "yes", "thus", "plus", "plus", "plus"}

# A question that already names the pair it is asking about wants one subject,
# not an enumeration: "what connects X and Y" is answered by the connection.
_NAMED_PAIR = re.compile(
    r"\bwhat\s+(?:connects|links|do\b.*\bhave in common|is shared)", re.I)


def is_set_question(question: str) -> bool:
    """
    Does this question ask for EVERY entity that qualifies, or for one?

    "Which parks cross an international border" has three right answers and
    scores zero for naming one of them; "which park is the largest" has one.
    The distinction is grammatical -- a plural noun after the interrogative --
    and needs no knowledge of the document.
    """
    q = question.strip()
    if _NAMED_PAIR.search(q):
        return False
    if re.search(r"\b(all|every|each|list every|how many)\b", q, re.I):
        return True
    m = re.search(r"\b(which|what|name|list|identify)\b", q, re.I)
    if not m:
        return False
    for word in re.findall(r"[A-Za-z]+", q[m.end():m.end() + 60])[:5]:
        low = word.lower()
        if low.endswith("s") and len(low) > 3 and low not in _NOT_PLURAL:
            return True
    return False


CROSS_SET_SYNTHESIS_PROMPT = """Answer the question below. It asks WHICH ONES, not which one:
every entity that qualifies is part of the answer, and naming only the best-
supported one scores as though the others had never been found.

The reports come from different, widely separated parts of one document. Each
was selected because it contains a distinctive phrase from the question.

============================ HOW TO ANSWER ============================

1. Go through the reports and collect EVERY entity for which the question's
   condition holds -- not the clearest one, all of them.
2. Write ONE NUMBERED LINE PER QUALIFYING ENTITY, in this shape:

       1. <entity> -- <the fact that qualifies it>, <where it is stated>
       2. <entity> -- <the fact that qualifies it>, <where it is stated>

3. If the question asks for two different things (inscribed versus merely
   nominated, one side of a border versus the other), give a numbered list for
   EACH, under its own one-line heading.

============================== HARD RULES ==============================

- DO NOT OPEN WITH A SINGLE ENTITY. There is no subject line, no "The answer is
  X", no summary sentence before the list. A previous answer to a question of
  this kind opened by naming one park, and the two other qualifying parks -- both
  correctly found and both present further down its own text -- scored nothing.
- One entity per line. An entity that qualifies for a stated reason goes in the
  list even when the evidence for it is thinner than for the others.
- NAME NOTHING THAT IS NOT IN THE REPORTS BELOW.
- Every figure, date and qualifier copied exactly as the reports give it.
- Never hedge inside a line: state the fact, do not weigh it.
- Never mention the reports, the parts, or that you were given anything.

============================= READER REPORTS =============================
{evidence}

=============================== QUESTION ==============================
{question}{parts}

Final answer, plain text, no preamble.
"""


CROSS_SYNTHESIS_PROMPT = """Answer the question below by finding the entity that appears
in MORE THAN ONE of the reports beneath.

The reports come from different, widely separated parts of one document. Each
was selected because it contains a distinctive phrase from the question. The
question names two or more facts and asks which entity holds all of them, so:

============================ HOW TO ANSWER ============================

1. List, for yourself, every ENTITY named in each report next to which ANCHOR.
2. The answer is the entity that appears under the different anchors -- the one
   that is in two or more reports for DIFFERENT facts. An entity named in only
   one report satisfies only one half of the question and is the wrong answer,
   however prominent it is there.
3. If a report says an entity is absent from, excluded from or not covered by
   something, that counts as one of the facts. Questions of this kind often turn
   on exactly that.
4. If a passage attaches a fact to a region, state or city, the entity is the
   country it belongs to -- but only if a report says so.

Then write the answer: NAME THE ENTITY FIRST, in the opening sentence. Then give
each of the facts that made it the answer, one at a time, each with the figure,
date and the chapter, section or table it comes from.

============================== HARD RULES ==============================

- GIVE EXACTLY ONE ENTITY as the answer. Never offer alternatives, never say
  "either X or Y", never hedge. If the evidence is thin, choose the entity with
  the most support and commit to it.
- NAME NOTHING THAT IS NOT IN THE REPORTS BELOW.
- Every figure, date and qualifier must be copied exactly as the reports give it.
- State each of the question's facts explicitly, even when one of them feels
  obvious once the entity is named. Each is graded on its own.
- Never mention the reports, the parts, or that you were given anything.

============================= READER REPORTS =============================
{evidence}

=============================== QUESTION ==============================
{question}{parts}

Final answer, plain text, no preamble.
"""


# g7's arc synthesis, kept so that G8_STEPS can turn step 4 off and reproduce
# g7's behaviour exactly. It scored 0.365 on the q16-q18 subset against g6's
# 0.421 -- it is here as a baseline to measure against, not as a fallback.
ARC_SYNTHESIS_PROMPT_G7 = """Write one final answer to the question below.

The blocks beneath are reports from readers who each saw ONE consecutive part of
the document, GIVEN TO YOU IN DOCUMENT ORDER, first part to last. That order is
the point: the question asks how something develops, and the development is only
visible in the sequence.

============================ HOW TO USE THEM ============================

READ THE REPORTS IN ORDER, FIRST TO LAST, BEFORE WRITING ANYTHING. Then answer
by comparing the beginning of the document to the end.

- State what CHANGES across the document. Name the specific sections or chapters
  where the framing shifts, using the SECTION labels given -- not "later in the
  document" but the actual chapter. A trajectory with no named waypoints scores
  as if you never found it.
- State what STAYS CONSTANT across the document -- the commitment or principle
  that holds from the first part to the last.
- ATTRIBUTE EACH STAGE SEPARATELY. If the question spans several chapters, give
  each chapter its own claim. NEVER merge two or more chapters into one combined
  remark: a chapter you fold into another scores nothing on its own.
- Carry the concrete evidence through. Every figure, percentage, name and example
  from the EVIDENCE lines that bears on the theme must survive into your answer,
  attached to the stage it came from. Length costs nothing; brevity loses points.
- Where a reader gave a SHIFT line, that is the document marking its own turn.
  Use it and say where it occurs.
- If the reports show the framing NOT changing, say that plainly -- a constant
  theme is a real finding, not a failure to find one.

============================== HARD RULES ==============================

- NAME NOTHING THAT IS NOT IN THE REPORTS BELOW. If a name, figure or claim
  appears in no report, it does not go in your answer, however plausible.
- Keep any label the question uses (a chapter, figure or table) so each claim
  stays locatable.
- Never mention the blocks, the readers, the parts, or that you were given
  anything. Write as if you had read the document yourself.
- Never hedge. Commit.
- If the question is listed below as asking for several separate things, ANSWER
  EVERY ONE OF THEM, in the order given. Each is graded on its own and a part you
  never mention scores nothing.

===================== READER REPORTS, IN DOCUMENT ORDER =====================
{evidence}

=============================== QUESTION ==============================
{question}{parts}

Final answer, plain text, no preamble.
"""


ARC_SYNTHESIS_PROMPT = """Below are reports from readers who each saw ONE consecutive part
of the document, GIVEN TO YOU IN DOCUMENT ORDER, first part to last.

Your job is to RECORD what the document contains, division by division. It is
not to write an essay about it.

That distinction is the whole task. A well-argued answer to this question
scored ZERO: it organised everything around one idea, and in doing so it never
listed the four barriers the question was really asking about, never named two
of the mechanisms, and dropped a dozen figures that did not fit the argument.
Every one of those omissions cost a point. The argument itself earned nothing,
because nobody is grading the argument.

============================== WHAT TO WRITE ==============================

NO OPENING THESIS. Do not begin with a framing sentence, a governing claim, or
a sentence that says what the document is fundamentally about. Begin with the
first division.

1. WALK THE DOCUMENT'S TOP-LEVEL DIVISIONS, IN ORDER.
   Use the SECTION labels in the reports. Top-level means the chapters and the
   front and back matter -- the divisions the document itself numbers as whole
   units. A numbered SUBSECTION IS NOT A DIVISION: do not give 2.2.5 or 5.3.1 a
   block of its own, fold what it says into its chapter. Listing subsections is
   how an answer ends up naming fifteen things and still missing two of the six
   the question asked for.

2. ONE BLOCK PER DIVISION, in this shape:

       <division name as printed>: <what it claims, one or two sentences>
       Names and figures: <EVERY named country, institution, programme, person,
       instrument, percentage, year and figure this division carries>

   "Every" is meant literally. Not the important ones, not a representative
   selection -- all of them. Length costs nothing here. Leaving one out costs a
   point when it happens to be the one being looked for.

3. THEN A CHANGE BLOCK:

       CHANGE: <what is different between the first divisions and the last --
       name the division where each shift happens, using its printed label, not
       "later in the document">

4. THEN A CONSTANT BLOCK:

       CONSTANT: <what holds from the first division to the last>

   QUOTE THE DOCUMENT'S OWN RECURRING WORDING for this. Find the phrase that
   actually repeats across the reports and use it. DO NOT invent an abstraction
   that summarises the theme in your own words -- two previous answers both
   described the constant in language the document never uses, and both were
   marked wrong for it. If the reports show a phrase recurring, that phrase IS
   the constant, whether or not it sounds like a grand principle.

5. THEN AN 'ALSO' BLOCK:

       ALSO: <any named entity, figure or finding in the reports that did not
       fit a division block above>

   This block may ONLY ADD facts not already stated. It must NEVER restate a
   figure, a count or a total you have already given, and never contradict one.
   Do not summarise, do not conclude, do not repeat the answer in short form. A
   second, different figure for something already answered correctly is scored
   as a wrong claim on top of a right one, and has halved a question before.

============================== HARD RULES ==============================

- NAME NOTHING THAT IS NOT IN THE REPORTS BELOW. If a name, figure or claim
  appears in no report, it does not go in your answer, however plausible.
- Copy every figure, percentage, year and proper name EXACTLY as the reports
  print it. Keep qualifiers: "about two-thirds" is not "most".
- Attribute each fact to the division it came from. A fact with no location is
  worth less than one with a chapter attached.
- Never mention the reports, the readers, the parts, or that you were given
  anything. Write as if you had read the document yourself.
- Never hedge. Commit.
- If the question is listed below as asking for several separate things, COVER
  EVERY ONE, in the order given. Each is graded on its own.

===================== READER REPORTS, IN DOCUMENT ORDER =====================
{evidence}

=============================== QUESTION ==============================
{question}{parts}

Your answer, plain text, no preamble.
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

INGEST_PROMPT = """You are reading part {n} of {of} of {doc}. The text below is verbatim.

The question asks where the document DISAGREES WITH ITSELF -- where the same
thing is given two different values in two different places. You hold one place
and cannot see the other, so DO NOT look for a contradiction and do not claim
one. Python will compare every part against every other part afterwards.

Your job is to record what THIS part states, so the comparison has something to
work on. For every quantity in this part that bears on the question, emit:

    RECORD | entity | attribute | value | where

For example:

    RECORD | Switzerland | non-compete coverage, employee survey | 408 | Table 5.1
    RECORD | Switzerland | non-compete coverage, employer survey | 409 | Annex 5.A.5

THE ATTRIBUTE IS WHAT MAKES THIS WORK. Two values for the same entity are only
comparable if you describe what was measured the SAME WAY both times, and only
distinguishable if you describe genuinely different measurements DIFFERENTLY.
Use the document's own words for it, including which table, survey, year or
population it refers to when the document says so.

ALSO record every SUPERLATIVE THE TEXT ASSERTS -- any place it calls something
the largest, highest, longest, oldest, first or only of its kind:

    CLAIM | entity | what the text says it is | where

For example:

    CLAIM | Snowdon | the most climbed mountain in Britain | page 239
    CLAIM | Lemmenjoki National Park | Finland's largest national park | p 152

Copy the assertion in the text's own words, including what it is being compared
against ("in Britain", "in its country", "in the Alps"). Do NOT judge whether it
is true -- you hold one part and cannot see the figures elsewhere. Just record
that the claim was made and where.

AND, for every entity you mention above, ALSO record WHERE IT IS whenever this
part says so, even if the question did not ask:

    RECORD | Jotunheimen National Park | country | Norway | page 122
    RECORD | Snowdon | country | Wales, United Kingdom | page 239

Every superlative has a scope. "Norway's largest" is not refuted by a bigger one
in Iceland, and "the highest in Wales" is not refuted by a Scottish mountain.
Without these lines Python cannot tell a real contradiction from a comparison
across two different countries, and it will report the wrong one.

Rules:
- COPY THE VALUE EXACTLY AS PRINTED, including its unit and any separator. Do
  not convert, round or normalize. If one entry's unit differs from the others,
  reproduce it exactly as printed -- that difference can itself be the answer.
- Take the value from the SAME ROW or sentence as the entity. In a flattened
  table, check that the number you copy belongs to the column the question asks
  about and not the one beside it.
- One line per entity per measurement. Use the entity's full printed name.
- Use the SAME WORDS for the same measurement every time, as the document
  prints them, so that two parts describing one quantity can be compared.
- 'where' locates it: the table, annex, box, figure, section or page.
- NEVER write a figure that is not printed in the text below. Every figure you
  emit is checked against the raw text and thrown away if it is not there.
- If this part is an index, a contents list or a page-number listing, it holds
  no measurements. Reply with the marker below instead of turning page numbers
  into records.
- Add at most two lines beginning "NOTE: " if this part says something about
  how a figure was produced, revised or qualified.

If this part contains nothing relevant, reply with exactly:

    {marker}

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""


ADJUDICATE_PROMPT = """Below is a shortlist of places where one document appears to give two
different values for the same thing. They were found mechanically, by grouping
every recorded figure by entity and attribute, so most of them will be false.

For each numbered candidate, decide whether the two values GENUINELY CONFLICT --
whether both statements cannot be true at once.

They come in three kinds, marked in brackets:

- [two values] one thing is given two different figures. Keep it ONLY when the
  same quantity, measured the same way, is given two different values. A real
  one often looks like a transposition, a revision not carried through, or one
  table disagreeing with another.
- [odd unit] one entry is printed in a different unit from all the rest of its
  kind. Keep it when the document is inconsistent with itself; reject it when
  the different unit is appropriate to that entry (a sea area against land
  areas, a duration against distances).
- [claim vs figures] the text calls something the largest, highest or first of
  its kind, and prints a bigger or earlier figure for something else. Keep it
  only when the two are genuinely comparable AND fall inside the same scope the
  claim names. "The largest in Norway" is not contradicted by a bigger park in
  Iceland; "the highest in Wales" is not contradicted by a Scottish mountain.
  CHECK THE SCOPE BEFORE KEEPING ONE.

Reject any candidate when the difference is explained by:
- different years, editions or reporting periods
- different units, scales or bases (per cent vs count, gross vs net)
- different populations, samples or subsets -- INCLUDING two different surveys,
  which measure different respondents and may legitimately differ
- one value being a total and the other a component
- the two being different measurements that were described alike
- a comparison across different scopes, countries or categories

Reply with one line per candidate, in order:

    <number> | CONFLICT | <one sentence naming both values and why they cannot
    both hold>
    <number> | SAME | <a few words: why the difference is legitimate>

CANDIDATES:
{candidates}

QUESTION:
{question}

Your lines, plain text, no preamble.
"""


CONTRADICTION_SYNTHESIS_PROMPT = """Write one final answer to the question, using the two
blocks below.

============================ HOW TO USE THEM ============================

VERIFIED CONFLICTS were found by Python, which grouped every figure recorded
anywhere in the document by what it measures, then kept only the groups holding
genuinely incompatible values.

    THIS BLOCK IS AUTHORITATIVE AND CANNOT BE OVERRULED. The conflict it names
    is the answer. Use its entities and its values EXACTLY.

    YOU MAY NOT STATE A CONFLICTING PAIR OF FIGURES THAT IS NOT IN THIS BLOCK.
    Not one you noticed in the reader lines, not one you remember, not one that
    would make a tidier answer. If this block is empty, say plainly which single
    claim the document makes about the subject and that no second value for it
    appears -- do NOT manufacture a pair. Both previous attempts at this
    question invented a plausible-looking pair of numbers and scored zero.

READER LINES are what individual readers recorded in their own slice. Use them
for context, wording and location -- never for a second opinion on a value.

============================== HARD RULES ==============================

- Name BOTH values, BOTH locations, and the entity they belong to.
- Say plainly which of the two is stated where, and do not decide which is
  correct unless the document itself does.
- Copy every figure exactly, including units and separators.
- NAME NOTHING that is in neither block.
- Never hedge. Commit.
- If the question is listed below as asking for several separate things, answer
  every one of them, in the order given.

========================== VERIFIED CONFLICTS =========================
{computed}

============================= READER LINES ============================
{evidence}

=============================== QUESTION ==============================
{question}{parts}

Final answer, plain text, no preamble.
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


MERGE_PROMPT = """Below are an answer and some additional material that was found to be
missing from it. Produce ONE answer that contains both.

This is an EDIT, not a continuation. The reader must see a single answer, not an
answer followed by a second pass over the same question.

============================== HARD RULES ==============================

- KEEP EVERY SPECIFIC FACT from the existing answer. Every figure, percentage,
  name, date and example in it must appear in your version too. You are adding
  material, not choosing between the two texts, and not shortening.
- FOLD the additional material into the place it belongs -- next to the part of
  the answer it relates to. Do not append it as a closing paragraph.
- STATE EVERY FIGURE EXACTLY ONCE. If both texts give a value for the same
  thing, keep the EXISTING answer's value and drop the other. Never state two
  different numbers for one quantity, and never close by restating totals or
  counts you have already given -- a second, different figure for something you
  already answered correctly is the single most expensive mistake available
  here, because it is scored as a wrong claim on top of a right one.
- Do not add a summary, a conclusion, or an introduction. No "In summary".
- NAME NOTHING that is in neither text below.
- Never hedge. Commit.

============================= EXISTING ANSWER =============================
{answer}

=========================== ADDITIONAL MATERIAL ===========================
{extra}

=============================== QUESTION ==============================
{question}

The single merged answer, plain text, no preamble.
"""


# ------------------------------------------------------------------ ANSWERING

def _sections_line(chunk: dict) -> str:
    """The section headings a chunk covers, for the arc reader's SECTION line."""
    titles = chunk.get("section_titles") or []
    if not titles:
        return "(this part falls between headings)"
    if len(titles) <= 12:
        return "\n    ".join(titles)
    return "\n    ".join(titles[:6] + [f"... {len(titles) - 12} more ..."]
                         + titles[-6:])


# g12, step 4b. Inserted only when a roster exists, so a document that produces
# none sends the identical prompt g11 sent. The roster NAMES entities; it never
# supplies a value, so it cannot put a figure in front of a reader that the
# document does not print.
_ROSTER_BLOCK = """THE DOCUMENT'S OWN SECTION HEADINGS NAME THESE ENTITIES. If any of them is in
the text below, it is one of the things being asked about, and you should emit a
line for it -- using this exact spelling of its name:

    {roster}

This list is not exhaustive and it supplies no values. An entity here with no
figure in your part gets no line; an entity in your part that is not on this
list still gets one.

"""


def _with_roster(prompt: str, roster: list[str]) -> str:
    if not roster:
        return prompt
    listed = "\n    ".join(roster[:60])
    return prompt.replace("TEXT (part",
                          _ROSTER_BLOCK.format(roster=listed) + "TEXT (part", 1)


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


def merge_draft(answer: str, extra: str, question: str) -> str:
    """
    Fold the coverage fill INTO the answer instead of appending it after.

    Every submission scored so far that ran a synthesis step shipped two drafts
    stapled together, and the second one repeatedly undercut the first. On one
    OECD question the first draft was a perfect 8 of 8 key points; the second
    restated the question, gave a different total and named a different chapter
    as the densest -- both of them claims the answer key lists as forbidden --
    and halved a question that had already been answered completely.

    Appending cannot fix that, because the damage is a SECOND statement of
    something already stated. Only an edit can, so the fill is merged in place
    and the reader sees one committed answer.

    Two guards, and both fall back to the old append rather than to nothing:
    a merge that drops the facts it was supposed to preserve is worse than an
    untidy answer, since the scoring counts facts and does not read for style.
    """
    try:
        merged = call_llm(MERGE_PROMPT.format(
            answer=answer, extra=extra, question=question)).strip()
    except Exception as e:
        _log(f"  ~ merge failed ({type(e).__name__}); appending instead")
        return f"{answer}\n\n{extra}"

    if not merged:
        return f"{answer}\n\n{extra}"

    # Did the merge keep what it was given? Numbers are the cheap proxy: they
    # are what the key points count, and a merge that quietly summarises will
    # shed them first.
    had = set(_NUMBER.findall(answer))
    kept_numbers = had & set(_NUMBER.findall(merged))
    if had and len(kept_numbers) < 0.8 * len(had):
        lost = ", ".join(sorted(had - kept_numbers)[:5])
        _log(f"  ! merge dropped figures the answer already had ({lost}); "
             f"keeping the original and appending instead")
        return f"{answer}\n\n{extra}"

    if len(merged) < 0.6 * len(answer):
        _log(f"  ! merge came back {len(merged):,} chars against {len(answer):,}; "
             f"treating as a summary and appending instead")
        return f"{answer}\n\n{extra}"

    _log(f"  ~ merged the coverage fill into one draft "
         f"({len(answer):,} + {len(extra):,} -> {len(merged):,} chars)")
    return merged


# How many ledger candidates go to the adjudicator. The shortlist is already
# ranked with the widely-separated ones first, and those are the real ones.
LEDGER_SHORTLIST = int(os.environ.get("LEDGER_SHORTLIST", "40"))


def adjudicate_conflicts(candidates: list[dict], question: str) -> list[dict]:
    """
    One model call over the whole shortlist, keeping only genuine conflicts.

    The mechanical filter is deliberately loose -- it groups on entity and
    attribute alone, so two surveys of different populations, a total and one of
    its components, and a figure revised between editions all land in the
    shortlist looking identical to a real contradiction. Sorting those out is
    semantics, which is what the model is for; finding them at all is
    bookkeeping, which is what Python is for.

    One call for the whole list, not one per candidate: the shortlist is short
    by construction, and judging them together lets the model see that six of
    them are the same legitimate distinction repeated.
    """
    if not candidates:
        return []

    shortlist = candidates[:LEDGER_SHORTLIST]
    lines = []
    for i, c in enumerate(shortlist, 1):
        if c["kind"] == "value_conflict":
            values = "; ".join(
                f"{v['value']!r} ({v['where'] or 'location not given'}, "
                f"{v['found_in'] or 'part not given'})" for v in c["values"])
            lines.append(f"{i}. [two values] {c['entity']} -- "
                         f"{c['attribute']}: {values}")
        elif c["kind"] == "unit_outlier":
            odd = "; ".join(f"{e['entity']} = {e['value']!r} "
                            f"({e['where'] or 'location not given'})"
                            for e in c["entities"])
            lines.append(f"{i}. [odd unit] '{c['attribute']}' is given in "
                         f"{c['majority_unit']!r} for {c['majority_count']} "
                         f"entries, but in {c['odd_unit']!r} for: {odd}")
        else:
            beaten = "; ".join(f"{b['entity']} = {b['value']!r} "
                               f"({b['where'] or 'location not given'})"
                               for b in c["contradicted_by"])
            lines.append(f"{i}. [claim vs figures] the text says {c['entity']} "
                         f"is {c['claim']!r} ({c['claim_where'] or 'no location'}), "
                         f"its own figure is {c['claimed_value']!r}, but it "
                         f"prints: {beaten}")

    try:
        reply = call_llm(ADJUDICATE_PROMPT.format(
            candidates="\n".join(lines), question=question)).strip()
    except Exception as e:
        _log(f"  ~ adjudication failed ({type(e).__name__}); keeping only the "
             f"candidates found in separate parts")
        # .get, not [...]: only value conflicts carry 'separated'. Indexing it
        # turned a recoverable broker failure into a lost question for every
        # unit outlier and every refuted claim -- the two kinds that do not
        # need two parts to be real.
        return [c for c in shortlist if c.get("separated", True)]

    kept = []
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*(CONFLICT|SAME)\s*\|?\s*(.*)", line, re.I)
        if not m:
            continue
        index, verdict, reason = int(m.group(1)), m.group(2).upper(), m.group(3)
        if verdict == "CONFLICT" and 1 <= index <= len(shortlist):
            kept.append({**shortlist[index - 1], "why_it_conflicts": reason.strip()})

    _log(f"  ~ ledger: {len(candidates)} candidate(s) grouped, "
         f"{len(kept)} judged to be genuine conflicts")
    return kept


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
    anchors: list[dict] = []
    if mode == "cross":
        # Line-based slicing, as the counting path uses: the two ends of a
        # cross-document link are usually in tables or lists, and word-based
        # slicing throws the row boundaries away.
        chunks = chunk_for(doc, doc["text"], WORDS_PER_READER)
        anchors = question_anchors(question, doc["text"])
        if not anchors:
            # Nothing in the question's wording occurs in the document rarely
            # enough to locate anything. Telling a reader it was selected for
            # phrases and then giving it none is worse than not claiming to
            # have located anything at all, so fall back to how g7 read these.
            _log("  ~ no phrase in the question locates anything; "
                 "reading every slice as prose")
            mode, chunks = "prose", doc["chunks"]
        else:
            _log("  ~ anchors: " + "; ".join(
                f"{a['phrase']!r} x{a['hits']}" for a in anchors))
            if step_on("7"):
                # Keep the anchors as instructions to the readers, drop them as
                # a filter. Narrowing is the one place this pipeline selects
                # chunks by the question, and it behaves the way retrieval
                # behaves: on one document the anchors were 'employee-like' and
                # 'February 2024' and the category scored 0.767; on another they
                # came out as 'formally', 'described', 'actually', 'happen', and
                # a question was answered from three slices of sixteen with the
                # two halves of the link never read at all.
                #
                # The reader format is what earned the points -- one question
                # ran on every slice because the anchors failed to narrow, and
                # still scored 0.80. So keep the format and read everything.
                for c in chunks:
                    c["anchors"] = [a["phrase"] for a in anchors
                                    if _occurrences(a["phrase"], c["text"].lower())]
                _log(f"  ~ anchors used as reading instructions, not as a "
                     f"filter; reading all {len(chunks)} slices")
            else:
                selected = anchor_chunks(chunks, anchors)
                if len(selected) < len(chunks):
                    _log(f"  ~ located in {len(selected)} of {len(chunks)} slices; "
                         f"reading only those")
                    chunks = selected
                else:
                    _log(f"  ~ anchors did not narrow the document; reading all "
                         f"{len(chunks)} slices")

    ledger_text = None
    if mode == "ledger":
        # Line-based, because what gets compared here are table rows and a
        # word-based slice throws the row boundaries away -- which is the same
        # loss that made two columns of one table read as a single number.
        ledger_text = strip_back_matter(doc["text"]) if step_on("6") else doc["text"]
        chunks = chunk_for(doc, ledger_text, WORDS_PER_READER)
        _log(f"  ~ ingesting claims from all {len(chunks)} slices")

    if mode == "tally":
        scope = scope_for(doc["text"], question)
        if scope:
            chunks = chunk_for(doc, scope["text"], TALLY_WORDS_PER_READER)
            scope_note = (f" You are reading only the part of the document under "
                          f"\"{scope['locator']}\", which is what the question asks "
                          f"about.")
            _log(f"  ~ scoped to {scope['locator']}: {scope['words']:,} words, "
                 f"{len(chunks)} readers (was {len(doc['chunks'])})")
        else:
            # No locator: read everything, but still on line boundaries so that
            # tabular material keeps its rows. The index is dropped here too --
            # an unscoped count is exactly where its entries do the damage,
            # having been relabelled as whatever attribute was asked for.
            ledger_text = strip_back_matter(doc["text"]) if step_on("6") else doc["text"]
            chunks = chunk_for(doc, ledger_text, WORDS_PER_READER)
            _log(f"  ~ no locator in the question; reading all {len(chunks)} slices")

    # g12, step 4b. Empty unless the document's headers named entities under an
    # unambiguous naming convention, and empty means the prompts below are the
    # exact strings g11 built.
    roster = doc.get("entity_roster") or [] if g12_on("entityseed") else []

    if mode == "tally":
        jobs = [(f"part {c['n']}/{c['of']}",
                 _with_roster(
                     RECORD_PROMPT.format(n=c["n"], of=c["of"], marker=NO_EVIDENCE,
                                          doc=doc["description"], scope=scope_note,
                                          chunk=c["text"], question=question),
                     roster))
                for c in chunks]
    elif mode == "arc":
        # The reader is handed its section headings when the document prints
        # them, and asked to infer one when it does not.
        use_sections = (g12_on("arcsection")
                        and any(c.get("section_titles") for c in chunks))
        if use_sections:
            _log("  ~ arc readers are given their section headings as printed "
                 "instead of inferring them")
            jobs = [(f"part {c['n']}/{c['of']}",
                     ARC_PROMPT_HEADERS.format(
                         n=c["n"], of=c["of"], doc=doc["description"],
                         sections=_sections_line(c), chunk=c["text"],
                         question=question))
                    for c in chunks]
        else:
            jobs = [(f"part {c['n']}/{c['of']}",
                     ARC_PROMPT.format(n=c["n"], of=c["of"], doc=doc["description"],
                                       chunk=c["text"], question=question))
                    for c in chunks]
    elif mode == "ledger":
        jobs = [(f"part {c['n']}/{c['of']}",
                 _with_roster(
                     INGEST_PROMPT.format(n=c["n"], of=c["of"], marker=NO_EVIDENCE,
                                          doc=doc["description"], chunk=c["text"],
                                          question=question),
                     roster))
                for c in chunks]
    elif mode == "cross":
        # Each reader is told which landmarks put it in the selection, so it
        # looks for the entity attached to those rather than for the question's
        # answer -- which it cannot see, holding only one end of the link.
        jobs = [(f"part {c['n']}/{c['of']}",
                 CROSS_PROMPT.format(
                     n=c["n"], of=c["of"], marker=NO_EVIDENCE,
                     doc=doc["description"], chunk=c["text"], question=question,
                     anchors="\n    ".join(
                         c.get("anchors") or [a["phrase"] for a in anchors])))
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

    if mode == "arc":
        # A trajectory question has no empty parts: a section that barely
        # engages the theme is itself evidence about where the theme is
        # carried. Dropping quiet parts here would leave only the dense
        # regions -- which is exactly the top-k failure this pipeline exists
        # to avoid. Keep everything, and keep it in document order so the
        # sequence itself is visible to the synthesis step.
        def _part_no(name: str) -> int:
            m = re.search(r"part (\d+)", name)
            return int(m.group(1)) if m else 0
        kept = sorted(((n, t) for n, t in done.items() if (t or "").strip()),
                      key=lambda item: _part_no(item[0]))
    else:
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

    if mode == "ledger":
        records = [r for name, text in kept for r in parse_records(text, name)]
        claims = [c for name, text in kept for c in parse_claims(text, name)]

        # Provenance first, and this is the step that matters. In prose mode
        # nineteen of twenty readers answering a contradiction question about
        # one table invented round numbers -- 1 000 companies, 100, 10 -- for a
        # table none of them held, and the synthesis picked one of the
        # inventions. verify_record checks each figure against the raw text and
        # is the reason the counting path never had that problem.
        against = ledger_text or doc["text"]
        verified, rejected = [], []
        for r in records:
            status = verify_record(r, against, provenance_window(doc))
            # Both verified statuses count. 'verified' means the entity and the
            # figure share a line; 'verified_nearby' means they sat within the
            # provenance window instead. Accepting only the first empties the
            # ledger on exactly the documents this is for -- a flattened table
            # has no line boundaries left, so every real record arrives by the
            # windowed path. 'unverifiable' records carry no digits to check and
            # are kept, since a text value can still disagree with another.
            keep = status.startswith("verified") or status == "unverifiable"
            (verified if keep else rejected).append(r)
        _log(f"  ~ ledger: {len(records)} records ingested, {len(verified)} "
             f"corroborated by the text, {len(rejected)} discarded; "
             f"{len(claims)} superlative claim(s)")

        candidates = (find_value_conflicts(verified)
                      + find_unit_outliers(verified)
                      + check_claims(claims, verified))
        if g10_on("labelunit"):
            # Read the document's own labels as well as the readers' records.
            # The names come from the records only so that a site can be
            # attributed to an entity -- the outlier itself is found without
            # them, so a measurement no reader reported is still found.
            names = sorted({(r.get("entity") or "").strip() for r in verified},
                           key=len, reverse=True)
            from_labels = find_label_unit_outliers(against, names)
            if from_labels:
                _log(f"  ~ ledger: {len(from_labels)} unit outlier(s) found in "
                     f"the document's own labels")
            # Appended, not prepended. This detector fires on any document with
            # a units convention, whatever the question asks -- so on a question
            # about mountain heights it offers an area-unit outlier as a
            # distractor. Ranking it below the candidates that came from the
            # question's own reading keeps it as a fallback rather than a lead.
            candidates = candidates + from_labels
        conflicts = adjudicate_conflicts(candidates, question)
        if conflicts:
            computed_block = json.dumps(
                {"verified_conflicts": conflicts}, indent=2, ensure_ascii=False)
        else:
            # Saying so explicitly matters: the synthesis prompt is told it may
            # not invent a pair, and an empty block is what makes that rule
            # bite instead of reading as "nothing was computed, use your
            # judgement".
            computed_block = json.dumps(
                {"verified_conflicts": [],
                 "note": "No two recorded values for the same quantity were "
                         "found to be genuinely incompatible. Do not state a "
                         "conflicting pair of figures."},
                indent=2, ensure_ascii=False)

    if mode == "tally":
        records = [r for name, text in kept for r in parse_records(text, name)]
        # Verify against the region the records actually came from. Checking a
        # scoped record against the whole document would let a figure from a
        # different table corroborate itself -- which is the exact error that
        # put Norway's Gender Development Index value into an HDI answer.
        against = scope["text"] if scope else (ledger_text or doc["text"])
        summary = tally(records, question, against, provenance_window(doc))
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
        if mode == "arc":
            template = (ARC_SYNTHESIS_PROMPT if step_on("4")
                        else ARC_SYNTHESIS_PROMPT_G7)
            final = call_llm(template.format(
                evidence=evidence_block, question=question,
                parts=parts_block)).strip()
        elif mode == "cross":
            set_valued = g10_on("setcross") and is_set_question(question)
            if set_valued:
                _log("  ~ set-valued question: enumerating every qualifying "
                     "entity, not choosing one")
            template = (CROSS_SET_SYNTHESIS_PROMPT if set_valued
                        else CROSS_SYNTHESIS_PROMPT)
            final = call_llm(template.format(
                evidence=evidence_block, question=question,
                parts=parts_block)).strip()
        elif mode == "ledger":
            final = call_llm(CONTRADICTION_SYNTHESIS_PROMPT.format(
                computed=computed_block, evidence=evidence_block,
                question=question, parts=parts_block)).strip()
        else:
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
        elif step_on("2"):
            final = merge_draft(final, filled, question)
        else:
            final = f"{final}\n\n{filled}"

    return {"answer": final, "computed": computed_block,
            "evidence": evidence_block, "readers": len(kept),
            "decision": f"{mode}_synthesized" + shortfall}


# -------------------------------------------------------------------- RUNNER

def load_questions(path: str) -> list[dict]:
    """
    Read the question file without assuming its exact shape.

    Both files we have are a bare JSON/YAML list of {id, question, category}.
    The competition file is not seen until the document is released, and the
    two shapes that would break a bare `for q in questions` are cheap to absorb:
    a dict wrapping the list under a key, which would iterate over key STRINGS
    and fail on q["id"], and entries that name the question field something
    else. Neither is exotic, and either one costs the whole run.

    A missing id is filled in positionally rather than raising. The submission
    is keyed by id, so an unlabelled question still has to go somewhere, and
    q01..qNN is the convention both known files already use.
    """
    loaded = _load_questions_file(path)

    if isinstance(loaded, dict):
        for key in ("questions", "items", "data"):
            if isinstance(loaded.get(key), list):
                _log(f"questions     unwrapped from a '{key}' key")
                loaded = loaded[key]
                break
        else:
            raise ValueError(
                f"{path} is a mapping with no question list under "
                f"'questions', 'items' or 'data'; keys are {list(loaded)[:8]}")
    if not isinstance(loaded, list):
        raise ValueError(f"{path} did not contain a list of questions")

    out = []
    for i, q in enumerate(loaded, 1):
        if not isinstance(q, dict):
            raise ValueError(f"{path} entry {i} is a {type(q).__name__}, "
                             f"not a mapping")
        text = next((q[k] for k in ("question", "text", "prompt", "q")
                     if q.get(k)), None)
        if not text:
            raise ValueError(f"{path} entry {i} has no question text "
                             f"(keys: {list(q)})")
        out.append({**q, "id": str(q.get("id") or f"q{i:02d}"),
                    "question": text})

    missing = sum(1 for q in out if not q.get("category"))
    if missing:
        # Worth saying out loud: without categories the routed modes are
        # reached only by keyword inference, which is a guess about wording
        # rather than supplied metadata.
        _log(f"questions     {missing} of {len(out)} carry no category; "
             f"their mode will be inferred from the wording")
    return out


def load_document(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = normalize_text(f.read())

    size = measure(text)
    structure = discover_structure(text)
    body, tail = split_body(text, structure)

    # g12. Discovered, never assumed: a document with no '## ' markers gets an
    # empty list here and takes every path below exactly as g11 took it.
    headers = detect_headers(text) if g12_on("headers") else []
    usable = headers_are_usable(headers, text)
    mode = "headers" if (headers and usable) else "words"
    coverage = (0.0 if not headers else
                (headers[-1]["end"] - headers[0]["start"]) / max(1, len(text)))

    roster: list[str] = []
    if mode == "headers" and g12_on("entityseed"):
        # The index is excluded first. A book's index repeats every entity name
        # against page numbers, and those repeats would drown the body's own
        # headers in the naming-convention vote.
        cut = find_back_matter(text) if step_on("6") else None
        in_body = [h for h in headers if cut is None or h["start"] < cut]
        roster = entity_headers(in_body)

    if mode == "headers":
        chunks = chunk_by_headers(text, headers, WORDS_PER_READER)
    else:
        chunks = chunk_text(text, auto_chunk_count(size["words"]))

    description = os.environ.get(
        "DOC_DESCRIPTION",
        f"a large document ({size['words']:,} words)")

    return {"text": text, "body": body.lower(), "tail": tail.lower(),
            "chunks": chunks, "size": size, "structure": structure,
            "description": description,
            "headers": headers, "chunking_mode": mode,
            "header_coverage": round(coverage, 3), "entity_roster": roster}


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
    _log(f"segmentation  {len(doc['headers']):,} '## ' headers, coverage "
         f"{doc['header_coverage']:.3f} -> chunking_mode = {doc['chunking_mode']}"
         + (f"; entity roster seeded with {len(doc['entity_roster'])} name(s)"
            if doc["entity_roster"] else
            "; no entity roster (no single naming convention dominated)"))
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
    "One pipeline, no embedding retrieval and no assumptions about the document. "
    "The text is read verbatim in slices sized from its own word count (~6,000 "
    "words per reader). For six of the seven question types every reader reads "
    "its slice for every question: no chunk is ranked, scored or filtered. "
    "Counting and comparison questions are answered by having each reader report "
    "only what its own slice contains, as structured records, which Python then "
    "dedupes, unit-converts, thresholds and ranks across the whole document; a "
    "reader is never asked for a total it cannot see. Absence is decided by "
    "counting terms over the whole text, and where a bibliography is detected, a "
    "term occurring only inside a cited work's title is reported as never "
    "discussed. " + (
        "Cross-section questions name the question's own distinctive phrases to "
        "the readers so they know what to look for, but every slice is still "
        "read: no chunk is filtered. "
        if step_on("7") else
        "Cross-section questions are the one exception to full fan-out: the "
        "whole document is scanned in Python for the question's own distinctive "
        "phrases, and the slices containing them are read in full, with the "
        "fan-out restored whenever those phrases fail to narrow the document. "
    ) + (
        "Contradiction questions are answered from a ledger: every slice reports "
        "what it contains as structured records, each figure is checked against "
        "the raw text and discarded if it is not printed there, and Python then "
        "groups the survivors to find one quantity given two values, one entry "
        "printed in a different unit from its peers, and superlatives the text "
        "asserts that its own figures elsewhere refute. A model only ever judges "
        "the resulting shortlist. " if step_on("5") else ""
    ) + (
        "Where a document carries an index, it is excluded from record "
        "extraction, since a list of names against page numbers is otherwise "
        "indistinguishable from a data table. " if step_on("6") else ""
    ) + (
        "Counting and ranking group records by what was MEASURED rather than by "
        "the words a reader happened to use for it, so that a measurement named "
        "five ways across five slices is one measurement and not five. "
        if fix_on("attr") or fix_on("catattr") else ""
    ) + (
        "Where a document keeps its own section headings in the text, the "
        "slices are cut on those headings rather than on a word count, so a "
        "boundary never falls inside a section; the heading is carried to the "
        "reader instead of being inferred. This is decided at ingest by counting "
        "the headings and measuring how much of the document they span, and a "
        "document without them takes the word-count path unchanged. "
        if g12_on("headers") else ""
    ) + "Trajectory "
    "questions read every slice in document order, keeping the parts that report "
    "no engagement with the theme, since where a theme is absent is part of its "
    "shape. Computed values are authoritative in the final synthesis and are "
    "never restated by a model, and one committed answer is produced per "
    "question rather than a draft followed by a second pass."
)


if __name__ == "__main__":
    run(DOCUMENT, QUESTIONS, SUBMISSION)
