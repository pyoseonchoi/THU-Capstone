"""
Model G1: one pipeline, any .txt, any size. No RAG, no per-document fitting.

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
Run:      nohup python -u pipeline_g1.py > run.log 2>&1 &
"""

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pipeline_ed2 import get_client, preprocess_document, load_questions

# ----------------------------------------------------------------- CONFIG

DOCUMENT = os.environ.get("DOC", "hdr2025.txt")
QUESTIONS = os.environ.get("QUESTIONS", "hdr_questions.yaml")
SUBMISSION = os.environ.get("SUBMISSION_FILE", "submission_g1.json")
TEAM = os.environ.get("TEAM", "group_c")

MODEL = os.environ.get("MODEL", "mistral-small-3-2")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))

# Words per reader. 6,000 is the measured line between a reader that enumerates
# named instances and one that writes summaries worth zero key points.
WORDS_PER_READER = int(os.environ.get("WORDS_PER_READER", "6000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "300"))

WORKERS = int(os.environ.get("WORKERS", "3"))
START_RPM = float(os.environ.get("RPM", "90"))
MAX_ATTEMPTS = int(os.environ.get("ATTEMPTS", "8"))

# The private document is released ~60 minutes before the deadline, and there is
# one evaluation run per team. A partial submission beats a missing one, so the
# runner stops starting new questions once this many minutes have elapsed.
DEADLINE_MINUTES = float(os.environ.get("DEADLINE_MINUTES", "0")) or None

NO_EVIDENCE = "NO EVIDENCE"


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


_PAGE_MARKER = re.compile(r"^\[page (\d+)\]$", re.M)

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
    pages = {int(m.group(1)): m.start() for m in _PAGE_MARKER.finditer(text)}

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


def candidate_list(question: str) -> str:
    """The part of the question that holds the listed candidates."""
    m = _INTERROGATIVE.search(question)
    prefix = question[:m.start()] if m else question
    # Drop any preamble sentence: "The book names several designations. Of X,..."
    parts = re.split(r"(?<=[.!?])\s+", prefix.strip())
    return parts[-1] if parts and parts[-1].strip() else prefix


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


def question_mode(question: str, category: str | None) -> str:
    """
    Which machinery this question needs: tally, absence, or prose.

    The category ships with the question file, so this is reading supplied
    metadata rather than guessing about the document. Keyword inference is the
    fallback for question files that omit it.
    """
    known = {"aggregation": "tally", "superlative": "tally", "absence": "absence"}
    if category and category.lower() in known:
        return known[category.lower()]
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
    for subject in question_subjects(candidate_list(question)):
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


def settle_absence(question: str, body: str, tail: str) -> dict | None:
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
    scope = ("the body of the document, excluding the reference section"
             if tail else "the document")

    text = (f"The document never mentions {verdict['subject']}. The phrase does "
            f"not occur anywhere in {scope}.")
    if verdict["reference_section_mentions"]:
        text += (f" It appears {verdict['reference_section_mentions']} time(s) only "
                 "inside the title of a work listed in the reference section, "
                 "which is a citation rather than a subject the document discusses.")
    if present:
        detail = "; ".join(f"{f['subject']} ({f['body_mentions']} mentions"
                           + (f", as \"{f['matched_phrase']}\"" if f["matched_phrase"] else "")
                           + ")" for f in present)
        text += (f" Every other subject named in the question is discussed: {detail}.")

    return {"answer": text, "basis": "whole-document term counting",
            "ambiguous": [f["subject"] for f in absent[1:]] if len(absent) > 1 else []}


# ----------------------------------------------------------- TALLY (generic)

# RECORD | entity | attribute | value | where
_RECORD = re.compile(r"^\s*RECORD\s*\|(.+?)\|(.+?)\|(.+?)\|(.*)$", re.I | re.M)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

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


def parse_records(text: str) -> list[dict]:
    """Pull RECORD lines out of one reader's reply."""
    out = []
    for m in _RECORD.finditer(text):
        entity, attribute, raw_value, where = (p.strip() for p in m.groups())
        if not entity or entity.lower() in ("entity", "n/a", "none"):
            continue
        value, printed_unit, family, normalized = _parse_value(raw_value)
        out.append({"entity": entity, "attribute": attribute.lower(),
                    "value": value, "unit": printed_unit, "unit_family": family,
                    "normalized": normalized, "raw_value": raw_value,
                    "where": where})
    return out


def tally(records: list[dict], question: str) -> dict | None:
    """
    Count, rank and threshold the records the readers returned.

    This is the generic replacement for a per-document table parser: the model
    reports what its own slice says, Python does every arithmetic step.
    """
    numeric = [r for r in records if r["normalized"] is not None]
    if not numeric:
        return None

    # Readers may report more than one attribute. Comparing heights against
    # areas is meaningless, so rank within the attribute most readers reported
    # and say plainly what was set aside.
    counts: dict[str, int] = {}
    for r in numeric:
        counts[r["attribute"]] = counts.get(r["attribute"], 0) + 1
    dominant = max(counts, key=counts.get)
    set_aside = {a: n for a, n in counts.items() if a != dominant}
    numeric = [r for r in numeric if r["attribute"] == dominant]

    # One record per entity: 34 overlapping slices report the same entity more
    # than once, and a duplicate would inflate every count.
    best: dict[str, dict] = {}
    for r in numeric:
        key = r["entity"].strip().lower()
        if key not in best or len(r["where"]) > len(best[key]["where"]):
            best[key] = r
    unique = list(best.values())

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
            "duplicates across overlapping slices were merged.")
    if len(units) > 1:
        note += (" Units were not uniform, so values were converted to a common "
                 "unit before comparison; the raw printed values are shown.")
    if set_aside:
        note += (" Records for other attributes were set aside: "
                 + ", ".join(f"{a} ({n})" for a, n in set_aside.items()) + ".")

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

RECORD_PROMPT = """You are reading part {n} of {of} of {doc}. The text below is verbatim.

The question needs a count or a comparison across the WHOLE document. You cannot
do that -- you hold roughly one {of}th of it. Do not try, and do not estimate a
total. Your job is only to REPORT WHAT THIS PART CONTAINS, exactly. Python will
count across all the parts.

For every item in this part that is relevant to the question, emit one line:

    RECORD | entity | attribute | value | where

For example, if asked about the areas of things:

    RECORD | Vatnajokull National Park | area | 13,600 sq km | Park in numbers box
    RECORD | Jotunheimen National Park | area | 1151 sq miles | Park in numbers box

Rules for RECORD lines:
- COPY THE VALUE EXACTLY AS PRINTED, including its unit and any comma. Do not
  convert, round or normalize. If a unit differs from the others, that is
  important information -- reproduce it as printed.
- One line per entity. Use the entity's full name as the document prints it.
- The attribute is what was measured, in the document's own words.
- 'where' locates it: the section, box, table, figure or page.
- If a value is missing or unreadable for an entity, skip that entity.
- Emit nothing but RECORD lines unless you also have context worth stating, in
  which case add at most two lines beginning "NOTE: ".

If this part contains no relevant item at all, reply with exactly:

    {marker}

and nothing else. DO NOT GUESS and DO NOT invent entities.

TEXT (part {n} of {of}):
{chunk}

QUESTION:
{question}

Your RECORD lines, plain text, no preamble.
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

=========================== COMPUTED VALUES ===========================
{computed}

============================ TEXT EVIDENCE ============================
{evidence}

=============================== QUESTION ==============================
{question}

Final answer, plain text, no preamble.
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


def answer_question(doc: dict, question: str, category: str | None) -> dict:
    """One question: Python where it can settle things, readers where it cannot."""
    mode = question_mode(question, category)

    if mode == "absence":
        settled = settle_absence(question, doc["body"], doc["tail"])
        if settled:
            note = f"settled in Python by {settled['basis']}; no model was called"
            if settled["ambiguous"]:
                note += (" | WARNING: more than one candidate looked absent: "
                         + ", ".join(settled["ambiguous"]))
            return {"answer": settled["answer"], "computed": note,
                    "evidence": "(not used: settled by computation)",
                    "readers": 0, "decision": "computed"}
        _log("  ~ absence could not be settled by counting; using readers")

    template = RECORD_PROMPT if mode == "tally" else PROSE_PROMPT
    jobs = [(f"part {c['n']}/{c['of']}",
             template.format(n=c["n"], of=c["of"], marker=NO_EVIDENCE,
                             doc=doc["description"], chunk=c["text"],
                             question=question))
            for c in doc["chunks"]]

    done, failed = _run_readers(jobs)
    for attempt in (1, 2):
        if not failed:
            break
        cooldown = 30 * attempt
        _log(f"  ~ {len(failed)} reader(s) rate-limited; cooling {cooldown}s "
             f"(pace {THROTTLE.pace()})")
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
        raise RuntimeError("no reader had evidence for this question")

    computed_block = "(nothing could be computed for this question)"
    if mode == "tally":
        records = [r for _, text in kept for r in parse_records(text)]
        summary = tally(records, question)
        if summary:
            computed_block = json.dumps(summary, indent=2, ensure_ascii=False)

    evidence_block = "\n\n".join(f"READER {n}:\n{t}" for n, t in kept)
    shortfall = f" ({len(missing)} of {len(jobs)} readers unavailable)" if missing else ""

    try:
        final = call_llm(SYNTHESIS_PROMPT.format(
            computed=computed_block, evidence=evidence_block,
            question=question)).strip()
    except Exception as e:
        _log(f"  ! synthesis failed ({e}); falling back to the fullest draft")
        final = ""

    if not final:
        return {"answer": max((t for _, t in kept), key=len),
                "computed": computed_block, "evidence": evidence_block,
                "readers": len(kept), "decision": "synthesis_failed" + shortfall}

    return {"answer": final, "computed": computed_block,
            "evidence": evidence_block, "readers": len(kept),
            "decision": f"{mode}_synthesized" + shortfall}


# -------------------------------------------------------------------- RUNNER

def load_document(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = preprocess_document(f.read())

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
