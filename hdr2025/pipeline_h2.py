"""
Model H2: the computed values stop passing through a language model.

WHY THIS EXISTS
---------------
H1 scored ~0.80 on the HDR practice set. Reading its own submission back
against the raw text shows that every one of its lost points came from the same
place, and it is not a reasoning failure -- it is a transcription failure.

H1 computes a value in Python, hands it to a 24B model, and asks that model to
retype it. Two stages do this, and both corrupted the value:

  1. The INDEX READER is given the store and asked to answer from it in prose.
     On h08 the store's term check contained exactly one absent phrase,
     "digital twins". The reader answered "The UN Global Digital Compact" --
     a phrase that occurs 6 times in the report body. Pure invention, from a
     store that held the right answer and nothing else.

  2. The SYNTHESIS is handed that prose under a heading saying THIS BLOCK IS
     AUTHORITATIVE AND CANNOT BE OVERRULED, and it overruled it anyway:

       h01  computed 74/50/43/26  ->  answered 17/58/62/20
       h02  computed 19/12/6      ->  answered 13/6/7, adding "the computed
                                      values ... do not override the specific
                                      counts from the Contents"
       h05  computed San Marino 85.7 -> answered Hong Kong 85.5, adding "the
                                      computed value ... is not included"

     The model announced that it was breaking the rule, in the answer text.

A prompt cannot fix this. Instruction-following is not a guarantee, and the
project's own rule already says so: anything countable is computed in Python
and the LLM only does what regex cannot. H1 applied that rule to the
CALCULATION and then handed the RESULT back to the model. H2 applies it to the
result as well -- if Python can settle the question, Python writes the answer
and no model is called for it at all.

WHAT CHANGES
------------
`settle()` inspects the compiled store and the question and, when the store
contains the answer outright, returns finished answer text. That covers
counting, superlative and absence questions -- 9 of the 21 here. Everything
else takes H1's path unchanged: 34 verbatim readers and the two-block
synthesis, which is where the model earns its place and where H2 changes
nothing.

`settle()` is driven by the SHAPE OF THE STORE, not by question ids, so a new
document gets the same treatment for whatever its compiler manages to parse,
and returns None -- falling back to H1 -- for everything else. That fallback is
recorded as `decision: "synthesized"` against `decision: "computed"`, so which
path produced an answer is always visible in the submission.

VERIFIED BEFORE WRITING (no LLM, against hdr2025.txt)
-----------------------------------------------------
  bands       74 / 50 / 43 / 26, total 193, ranks contiguous 1-74, 75-124,
              125-167, 168-193, and a recount straight off the report's own
              published HDI cutoffs agrees exactly.
  h05         printed row 29 is "San Marino ... 85.7"; Hong Kong is 85.5.
              H1 answered Hong Kong. The compiler was right.
  h08         "digital twins" 0 in body and 0 in references; "global digital
              compact" 11 in body. H1 inverted both halves.

Run:
    nohup python -u pipeline_h2.py > run.log 2>&1 &
"""

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pipeline_h1 as H1
from pipeline_h1 import _STOPWORDS, build_store
from pipeline_ed2 import get_client, preprocess_document, load_questions
from pipeline_v6 import (ANSWER_TEMPERATURE, MODEL_ANSWER, _dumps,
                         format_duration, timed)
from pipeline_v9 import NO_EVIDENCE, declares_no_evidence
import hdr_compile

DOCUMENT = os.environ.get("HDR_DOCUMENT", "hdr2025.txt")
QUESTIONS = os.environ.get("HDR_QUESTIONS", "hdr_questions.yaml")
DEFAULT_SUBMISSION = os.environ.get("SUBMISSION_FILE", "submission_h4.json")

# The broker rate-limits, and H1's answer to that was three retries at 1s, 2s
# and 4s with no jitter. Six worker threads hitting a per-minute quota all wake
# on the same schedule and collide again, so a question would burn its whole
# retry budget in seven seconds and go out built from 8 of its 34 readers.
# Failed calls are free, so the budget here is generous in time and the pacing
# is shared across threads rather than per-call.
WORKERS = int(os.environ.get("HDR_WORKERS", "3"))
START_RPM = float(os.environ.get("HDR_RPM", "90"))     # opening pace, adapts down
MAX_ATTEMPTS = int(os.environ.get("HDR_ATTEMPTS", "8"))

# Splits a question into the items it lists. Same expression H1 uses for its
# term check: cutting on conjunctions is what stops a phrase straddling two
# list items ("deepfakes algorithmic management" is absent only because it is
# two different subjects).
_SEGMENTS = re.compile(r"[,;:.?()—–]|\band\b|\bor\b|\bwhile\b|\byet\b|\bbut\b")
_WORDS = re.compile(r"[a-z][a-z'-]+")


def _fmt(value: float) -> str:
    """166812.0 -> '166,812'; 0.972 -> '0.972'; 85.7 -> '85.7'."""
    if value == int(value) and abs(value) >= 1000:
        return f"{int(value):,}"
    if value == int(value):
        return str(int(value))
    return str(value)


# ------------------------------------------------------- QUESTION SUBJECTS

def question_subjects(question: str) -> list[dict]:
    """
    The list items a question names, each with the content phrases inside it.

    "Of deepfakes, algorithmic management, care technologies and the metaverse"
    yields four subjects. Function words are dropped from the phrases used for
    matching but kept in the display name, and the display name keeps the
    question's own capitalization -- an answer that says "the un global digital
    compact" looks like something the pipeline made up.
    """
    subjects = []
    for segment in _SEGMENTS.split(question):
        words = _WORDS.findall(segment.lower())
        content = [w for w in words if w not in _STOPWORDS]
        if not content:
            continue

        # Every n-gram of the segment that begins and ends on a content word.
        # Cutting the segment AT its function words instead would have been the
        # obvious move and is wrong: "people with disabilities" and "social
        # dialogue on algorithmic management" are single subjects with a
        # function word inside them, and splitting there loses the phrase the
        # document actually prints.
        grams = set()
        for size in range(1, len(words) + 1):
            for i in range(len(words) - size + 1):
                gram = words[i:i + size]
                if gram[0] in _STOPWORDS or gram[-1] in _STOPWORDS:
                    continue
                grams.add(" ".join(gram))

        # Display: the segment as the question wrote it, minus the leading
        # connective the split left behind ("of deepfakes" -> "deepfakes").
        display = segment.strip().strip(",;:. ")
        display = re.sub(r"^(?:of|among|which|the following)\s+", "", display,
                         flags=re.I).strip()
        subjects.append({
            "display": display or " ".join(content),
            "content_words": len(content),
            # Longest by WORD COUNT first: the most specific phrase that occurs
            # is the honest evidence, not the most frequent one. "people" is a
            # poor witness for "people with disabilities".
            "phrases": sorted(grams, key=lambda g: (-len(g.split()), -len(g))),
        })
    return subjects


def _occurrences(phrase: str, text: str) -> int:
    """Count a phrase and its obvious singular/plural forms."""
    forms = {phrase}
    if phrase.endswith("ies"):
        forms.add(phrase[:-3] + "y")
    elif phrase.endswith("s"):
        forms.add(phrase[:-1])
    else:
        forms.add(phrase + "s")
    return sum(text.count(f) for f in forms)


def subject_coverage(question: str, body: str, tail: str) -> list[dict]:
    """
    For each subject the question lists, is it discussed in the report body?

    A subject counts as PRESENT when some phrase of two or more content words
    inside it occurs -- or, for a one-word subject, that word. Requiring two
    words is what stops "artificial intelligence audit protocols" from being
    called present merely because "intelligence" appears everywhere, while
    still recognizing "excessive screen time in early childhood" from "screen
    time".

    This replaces H1's check_question_terms for absence questions. H1 counted
    arbitrary n-grams and reported the absent ones; it got the right answer for
    h08 and the index reader then ignored it. Working from the question's own
    list items means the verdict is a comparison across candidates, which is
    what the question actually asks.
    """
    findings = []
    for subject in question_subjects(question):
        # A single word is evidence only when the subject IS a single word:
        # "intelligence" must not certify "artificial intelligence audit
        # protocols" as discussed.
        minimum = 1 if subject["content_words"] == 1 else 2
        candidates = [p for p in subject["phrases"] if len(p) >= 5
                      and sum(1 for w in p.split() if w not in _STOPWORDS) >= minimum]

        # Phrases come longest-first, so the first that occurs is the most
        # specific witness available.
        best_phrase, best_count = None, 0
        for phrase in candidates:
            count = _occurrences(phrase, body)
            if count:
                best_phrase, best_count = phrase, count
                break

        # The reference-section count must describe THE SAME phrase as the
        # verdict. Taking a maximum over sub-phrases produced "digital twins
        # ... appears 162 times in the reference section" -- a fabricated
        # number that contradicted the verdict in the sentence before it.
        full_phrase = candidates[0] if candidates else None
        in_tail = _occurrences(best_phrase or full_phrase, tail) \
            if (best_phrase or full_phrase) else 0

        findings.append({
            "subject": subject["display"],
            "matched_phrase": best_phrase,
            "full_phrase": full_phrase,
            "body_mentions": best_count,
            "reference_section_mentions": in_tail,
            "discussed_in_report": best_count > 0,
        })
    return findings


# -------------------------------------------------------------- SETTLEMENT

def _settle_absence(question: str, body: str, tail: str) -> dict | None:
    q = question.lower()
    if not re.search(r"never (mentioned|appears|discussed)|not mentioned anywhere", q):
        return None

    found = subject_coverage(question, body, tail)
    absent = [f for f in found if not f["discussed_in_report"]]
    present = [f for f in found if f["discussed_in_report"]]
    if len(found) < 2 or not absent:
        return None                       # not a list question, or nothing absent

    # One verdict, never alternatives. If more than one candidate looks absent
    # the longest name wins, but the ambiguity is recorded rather than hidden.
    verdict = max(absent, key=lambda f: len(f["subject"]))

    lines = [f"The report never mentions {verdict['subject']}. "
             f"The phrase does not occur anywhere in the body of the report "
             f"(pages 1-{hdr_compile.BODY_LAST_PAGE})"]
    if verdict["reference_section_mentions"]:
        lines[0] += (f"; it appears {verdict['reference_section_mentions']} time(s) "
                     "only inside the title of a work listed in the reference "
                     "section, which is a citation and not a subject the report "
                     "discusses")
    lines[0] += "."

    if present:
        detail = "; ".join(
            f"{f['subject']} ({f['body_mentions']} mentions"
            + (f", as \"{f['matched_phrase']}\"" if f["matched_phrase"] else "")
            + ")" for f in present)
        lines.append(f"Every other subject named in the question is discussed "
                     f"in the report: {detail}.")

    return {"answer": " ".join(lines), "basis": "body-only coverage index",
            "ambiguous": [f["subject"] for f in absent[1:]] if len(absent) > 1 else []}


def _settle_superlative(question: str, table1: dict) -> dict | None:
    q = question.lower()
    if "highest" not in q or not table1:
        return None

    fields = [("life expectancy", "highest_life_expectancy_2023", "years",
               "life expectancy at birth in 2023"),
              ("gross national income", "highest_gni_per_capita_2023",
               "(2021 PPP $)", "gross national income per capita in 2021 PPP dollars"),
              ("gni", "highest_gni_per_capita_2023", "(2021 PPP $)",
               "gross national income per capita in 2021 PPP dollars"),
              ("human development index", "highest_hdi_2023", "",
               "2023 Human Development Index value")]
    for keyword, key, unit, label in fields:
        if keyword not in q:
            continue
        top = table1.get(key)
        if not top:
            return None
        value = f"{_fmt(top['value'])}{' ' + unit if unit else ''}".strip()
        text = (f"Among the ranked entries in Statistical Annex Table 1, "
                f"{top['country']} has the highest {label}: {value}. "
                f"{top['country']} is ranked {top['rank']} in the table.")
        if top.get("tied_with"):
            text += f" Tied at the same value: {', '.join(top['tied_with'])}."
        else:
            text += " No other ranked entry reports a higher value, and there is no tie at the top."
        return {"answer": text, "basis": "Statistical Annex Table 1, all ranked rows"}
    return None


def _settle_group_counts(question: str, table1: dict) -> dict | None:
    q = question.lower()
    if not table1 or "how many" not in q:
        return None
    groups = table1.get("ranked_entries_by_hdi_group") or {}
    if not groups or not re.search(r"\bgroup", q):
        return None

    parts = ", ".join(f"{n} in the {band.lower()} group"
                      for band, n in groups.items())
    text = (f"In Statistical Annex Table 1 the ranked entries divide as follows: "
            f"{parts}. The total number of ranked entries is "
            f"{table1['ranked_entries_total']}, and the four groups sum to that "
            f"total exactly. The lowest printed rank is {table1['lowest_rank']}. "
            f"Unranked entries in the 'Other countries or territories' block are "
            f"excluded, since the question asks for ranked entries.")
    return {"answer": text, "basis": "Statistical Annex Table 1, all ranked rows"}


def _settle_contents(question: str, contents: dict) -> dict | None:
    q = question.lower()
    if not contents or "how many" not in q:
        return None

    named = [name for name in contents if name.lower() in q]
    if not named:
        return None

    # "how many figures belong to the Overview and to each of Chapters 1-6"
    per_section = re.search(r"chapter|overview|section", q)
    if per_section and len(named) == 1:
        section = contents[named[0]]
        breakdown = ", ".join(f"{k} {v}" for k, v in
                              section["count_by_section"].items())
        text = (f"The Contents lists {section['count']} items under {named[0]} "
                f"in total, including the {section['spotlight_items_included']} "
                f"whose identifiers begin with S. By section: {breakdown}. "
                f"The section with the most is {section['section_with_most']}, "
                f"with {max(section['count_by_section'].values())}.")
        return {"answer": text, "basis": "the Contents lists, pages 11-13"}

    counts = ", ".join(f"{contents[n]['count']} under {n}" for n in named)
    included = sum(contents[n]["spotlight_items_included"] for n in named)
    text = (f"The Contents lists {counts}. These totals include the {included} "
            f"entries whose identifiers begin with S.")
    return {"answer": text, "basis": "the Contents lists, pages 11-13"}


def settle(question: str, computed: dict, body: str, tail: str) -> dict | None:
    """
    Answer from the compiled store alone, or return None.

    Order matters: the per-section figure question and the group-count question
    both say "how many", so the more specific test runs first. Anything this
    function declines falls through to H1's readers, which is the floor and a
    decent one.
    """
    table1 = computed.get("annex_table1") or {}
    contents = computed.get("contents") or {}

    for attempt in (_settle_absence(question, body, tail),
                    _settle_superlative(question, table1),
                    _settle_contents(question, contents),
                    _settle_group_counts(question, table1)):
        if attempt:
            return attempt
    return None


# ------------------------------------------------------------- THE THROTTLE

def _is_rate_limit(error: Exception) -> bool:
    return (getattr(error, "status_code", None) == 429
            or "RateLimit" in type(error).__name__
            or "429" in str(error) or "rate limit" in str(error).lower())


class Throttle:
    """
    One shared pace for every thread, widened by 429s and narrowed by success.

    Per-call backoff cannot fix a per-minute quota: each thread backs off
    privately, wakes on the same 1/2/4 schedule as the others, and collides
    again. The limiter has to be global and it has to stay slow after a
    rejection rather than resetting on the next call.
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
        """A 429: halve the rate for everyone and report the cooldown to serve."""
        with self._lock:
            self.rejections += 1
            # Ceiling at 15s: with 3 workers that is still ~4 calls/minute, and
            # a question needs 35 of them. Slower than this and the run cannot
            # finish at all, so the retry passes take over instead.
            self._interval = min(15.0, max(self._interval * 2, 1.0))
            cooldown = self._interval
            # Push every thread's next slot out, not just this one's.
            self._next_at = max(self._next_at, time.monotonic() + cooldown)
        return cooldown

    def relax(self) -> None:
        # 20% per success, not 5%: at 5% a single storm leaves the run crawling
        # for the next seventy calls even though the broker has recovered.
        with self._lock:
            self._interval = max(self._floor, self._interval * 0.8)

    def pace(self) -> str:
        return f"{60.0 / self._interval:.0f} calls/min"


THROTTLE = Throttle(START_RPM)


def call_llm(prompt: str, model: str = MODEL_ANSWER,
             temperature: float | None = ANSWER_TEMPERATURE) -> str:
    """
    One broker call, paced globally and retried patiently on 429.

    Goes to the client directly rather than through pipeline_ed2.call_llm,
    whose own four-attempt loop would multiply against this one and flood the
    log with retries we cannot pace.
    """
    messages = [{"role": "user", "content": prompt}]
    sampling = {} if temperature is None else {"temperature": temperature}
    last = None

    for attempt in range(MAX_ATTEMPTS):
        THROTTLE.wait()
        try:
            response = get_client().chat.completions.create(
                model=model, messages=messages, **sampling)
            THROTTLE.relax()
            return response.choices[0].message.content or ""
        except Exception as e:
            last = e
            if not _is_rate_limit(e):
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise                       # permanent: bad key, prompt too big
                if attempt >= 2:
                    raise
                time.sleep(2 ** attempt + random.random())
                continue
            cooldown = THROTTLE.penalize()
            # Jitter matters as much as the wait: without it every thread
            # returns at the same instant and re-collides.
            time.sleep(cooldown * (1.0 + random.random()))

    raise last


# H1's prompts still go through its own module-level call_llm, so point that at
# this one too. Stated out loud rather than done quietly.
H1.call_llm = call_llm


# ------------------------------------------------------------ THE OVERRIDE

def _run_readers(jobs: list[tuple[str, str]]) -> tuple[dict, list]:
    """Run a batch of reader prompts; return what came back and what failed."""
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


def answer_question(views: dict, question: str) -> dict:
    """
    Python first; readers and the two-block synthesis for the rest.

    Reimplemented rather than delegated to H1 for one reason: H1 treats a
    reader that failed and a reader that had nothing to say as the same thing.
    Both end up dropped, so a question whose readers were rate-limited away
    goes out looking like a question nobody had evidence for -- h10 was
    answered from 8 of 34 slices in 16 seconds and nothing in the submission
    said so. Here a failure is retried after a cooldown, and whatever is still
    missing is named in the decision field.
    """
    settled = settle(question, views["computed"], views["body"], views["tail"])
    if settled:
        note = f"settled in Python from {settled['basis']}; no model was called"
        if settled.get("ambiguous"):
            note += (" | WARNING: more than one candidate looked absent: "
                     + ", ".join(settled["ambiguous"]))
        return {"answer": settled["answer"], "computed": note,
                "evidence": "(not used: this question is settled by computation)",
                "readers": 0, "decision": "computed"}

    store_view = dict(views["computed"])
    if H1.TERM_CHECK:
        store_view["question_term_check"] = H1.check_question_terms(
            question, views["body"], views["tail"])

    jobs = [("index", H1.INDEX_PROMPT.format(store=_dumps(store_view),
                                             marker=NO_EVIDENCE, question=question))]
    for chunk in views["chunks"]:
        jobs.append((f"part {chunk['n']}/{chunk['of']}", H1.CHUNK_PROMPT.format(
            n=chunk["n"], of=chunk["of"], marker=NO_EVIDENCE,
            frac=f"one {chunk['of']}th", chunk=chunk["text"], question=question)))

    done, failed = _run_readers(jobs)

    # A reader lost to a 429 is a slice of the report missing from the answer,
    # so it is worth waiting for. Two extra passes, each after a real cooldown.
    for attempt in (1, 2):
        if not failed:
            break
        cooldown = 30 * attempt
        print(f"  ~ {len(failed)} reader(s) rate-limited; cooling down {cooldown}s "
              f"then retrying (pace now {THROTTLE.pace()})")
        time.sleep(cooldown)
        retry_jobs = [(name, prompt) for name, prompt in jobs
                      if name in {n for n, _ in failed}]
        recovered, failed = _run_readers(retry_jobs)
        done.update(recovered)

    missing = [name for name, _ in failed]
    if missing:
        print(f"  ! {len(missing)} reader(s) never returned: {', '.join(missing)}")

    index_answer, raw = "", []
    for name, text in done.items():
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
    shortfall = f" ({len(missing)} of {len(jobs)} readers unavailable)" if missing else ""

    try:
        final = call_llm(H1.SYNTHESIS_PROMPT.format(
            computed=computed_block, evidence=evidence_block, question=question))
    except Exception as e:
        print(f"  ! synthesis failed ({e}); falling back to the fullest draft")
        return {"answer": index_answer or max((t for _, t in raw), key=len),
                "computed": computed_block, "evidence": evidence_block,
                "readers": len(raw) + bool(index_answer),
                "decision": "synthesis_failed" + shortfall}

    final = final.strip()
    if not final:
        return {"answer": index_answer or max((t for _, t in raw), key=len),
                "computed": computed_block, "evidence": evidence_block,
                "readers": len(raw) + bool(index_answer),
                "decision": "synthesis_empty" + shortfall}

    return {"answer": final, "computed": computed_block,
            "evidence": evidence_block, "readers": len(raw) + bool(index_answer),
            "decision": "synthesized" + shortfall}


# H1's run_submission calls the answer_question in its own module namespace, so
# the override is installed there. Stated out loud rather than done quietly.
# --------------------------------------------------------------- THE RUNNER

def _completed(path: str) -> dict:
    """
    Answers worth keeping from an earlier attempt at the same file.

    A rate-limited run is interrupted, not wasted. Anything that came back
    whole is reused; anything that fell back to a partial draft, or ran short
    of readers, is re-asked.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            previous = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    keep = {}
    for entry in previous.get("answers", []):
        decision = entry.get("decision", "")
        if (entry.get("answer") and not entry.get("error")
                and decision in ("computed", "synthesized")):
            keep[entry["id"]] = entry
    return keep


def run_submission(store: dict, document_text: str, questions: list[dict],
                   output_path: str, team: str = "group_c", notes: str = "",
                   ingest_seconds: float = 0.0) -> dict:
    with timed("build the reader views"):
        pages = hdr_compile.split_pages(document_text)
        last = hdr_compile.BODY_LAST_PAGE
        views = {
            "computed": store["computed"],
            "chunks": H1.chunk_raw_text(document_text),
            "body": "\n".join(pages[p] for p in sorted(pages) if p <= last).lower(),
            "tail": "\n".join(pages[p] for p in sorted(pages) if p > last).lower(),
        }

    reuse = _completed(output_path)
    if reuse:
        print(f"  resuming: {len(reuse)} answer(s) kept from {output_path} "
              f"({', '.join(sorted(reuse))})")
    print(f"  {len(views['chunks'])} raw readers, {WORKERS} workers, "
          f"opening pace {THROTTLE.pace()}")

    started = time.perf_counter()
    submission = {"team": team, "notes": notes, "answers": [],
                  "timing": {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "ingest_seconds": round(ingest_seconds, 2),
                             "raw_readers": len(views["chunks"])}}
    per_question, decisions = {}, {}

    for q in questions:
        qid = q["id"]
        if qid in reuse:
            submission["answers"].append(reuse[qid])
            decision = reuse[qid].get("decision", "reused")
            decisions[decision] = decisions.get(decision, 0) + 1
            print(f"  [   reused] {qid}")
            continue

        print(f"Answering {qid}...")
        entry, question_start = {"id": qid}, time.perf_counter()
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
        submission["timing"].update({
            "answering_seconds": round(time.perf_counter() - started, 2),
            "decisions": dict(decisions),
            "per_question_seconds": dict(per_question),
            "rate_limit_rejections": THROTTLE.rejections,
        })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"\nSubmission written to {output_path} "
          f"({len(submission['answers'])} answers)")
    print(f"  decisions: {decisions}")
    print(f"  429s absorbed: {THROTTLE.rejections}; final pace {THROTTLE.pace()}")
    incomplete = [a["id"] for a in submission["answers"]
                  if not a.get("answer") or a.get("decision", "").endswith(")")]
    if incomplete:
        print(f"  INCOMPLETE, re-run to finish: {', '.join(incomplete)}")
    return submission


NOTES = (
    "Anything the document settles by counting is answered in Python and never "
    "passes through a model: every ranked row of Statistical Annex Table 1, the "
    "Contents lists counted per section, and a coverage index that separates the "
    "report body (pages 1-233) from the reference section, so a term occurring "
    "only inside a cited work's title is correctly reported as never discussed. "
    "The remaining questions are answered by thirty-four readers who each hold a "
    "verbatim slice of the report and report the patterns and concrete instances "
    "it contains, combined by a synthesis that may name nothing appearing in no "
    "reader's lines. Every reader sees every slice for every question; no chunk "
    "is ever selected, ranked or filtered by the question."
)


if __name__ == "__main__":
    run_started = time.perf_counter()

    with timed("read and preprocess document"):
        with open(DOCUMENT, "r", encoding="utf-8") as f:
            document_text = preprocess_document(f.read())

    store = build_store(document_text)
    ingest_seconds = time.perf_counter() - run_started
    questions = load_questions(QUESTIONS)

    pages = hdr_compile.split_pages(document_text)
    last = hdr_compile.BODY_LAST_PAGE
    body = "\n".join(pages[p] for p in sorted(pages) if p <= last).lower()
    tail = "\n".join(pages[p] for p in sorted(pages) if p > last).lower()

    settled = [q["id"] for q in questions
               if settle(q["question"], store["computed"], body, tail)]
    print(f"\nsettled in Python : {len(settled)}/{len(questions)} "
          f"({', '.join(settled)})")
    print(f"sent to readers   : {len(questions) - len(settled)}")

    run_submission(store, document_text, questions, DEFAULT_SUBMISSION,
                   notes=NOTES, ingest_seconds=ingest_seconds)
    print(f"\ntotal: {format_duration(time.perf_counter() - run_started)}")
