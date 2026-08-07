"""
Steps 1 and 2 of g12, unwired: detect_headers() and headers_are_usable(),
run standalone against every document with score history.

Nothing here touches the pipeline. The point is to see the real counts and
coverage numbers on a header-rich document and on two header-absent ones
BEFORE any threshold is chosen, and before anything is wired into scoring.

    python3 -u header_probe.py
"""

import collections
import os
import re
import sys

# The pipeline's own normalizer, so the probe sees exactly the text the
# readers would see. The parks file stores its newlines as the literal two
# characters backslash-n; normalize_text restores them. Header detection must
# work either way, which is why the pattern below is not line-anchored.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "base", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.environ.get("BASE_PIPELINE", "pipeline_g12.py")))
G11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G11)


# --------------------------------------------------------------- STEP 1

# '## ' followed by a title. NOT line-anchored, and deliberately so: the parks
# file arrives as one line of 578,000 characters with literal backslash-n where
# the newlines were, so `^##` under re.M matches exactly nothing in it. The
# title runs to the next newline (real or literal), the next '#', or a length
# cap -- whichever comes first.
HEADER_MAX_TITLE = int(os.environ.get("HEADER_MAX_TITLE", "90"))

_HEADER = re.compile(
    r"##[ \t]+"                       # the marker and at least one space
    r"(?P<title>[^\n#]{1,%d}?)"       # the title, lazily, no newline and no '#'
    r"(?=\s*(?:\\n|\n|#|$))" % HEADER_MAX_TITLE)


def detect_headers(text: str) -> list[dict]:
    """
    Every '## Title' occurrence in the raw text, as ordered character spans.

    Returns [{'title', 'start', 'end'}], where 'start' is the offset of the
    '##' marker and 'end' is the next header's 'start' (len(text) for the
    last). The spans are therefore non-overlapping and cover the document from
    the first header onwards.
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


# --------------------------------------------------------------- STEP 2

def headers_are_usable(headers: list[dict], text: str,
                       min_count: int = 5,
                       min_coverage: float = 0.5) -> bool:
    """
    Are these headers a real partition of the document, or just a title page?

    Two tests, both cheap:
      - at least `min_count` of them, so a Contents page or a single heading
        cannot switch the whole pipeline onto a different chunker;
      - the span from the first header to the end of the last must be at least
        `min_coverage` of the document, so headers clustered in one region
        (a Contents listing that repeats every chapter title, then nothing for
        the rest of the book) do not qualify either.
    """
    if not text or len(headers) < min_count:
        return False
    covered = headers[-1]["end"] - headers[0]["start"]
    return covered >= min_coverage * len(text)


def coverage_of(headers: list[dict], text: str) -> float:
    if not headers or not text:
        return 0.0
    return (headers[-1]["end"] - headers[0]["start"]) / len(text)


# ---------------------------------------------------------------- PROBE

# Beside this file by default; DOCS=a.txt,b.txt overrides. A document that is
# not present is skipped with a note rather than crashing the probe.
HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = ["nationalparks_europe.txt", "oecd2026_fullscan_test.txt",
            "gem2024_5_fullscan_no_indent.txt", "hdr2025.txt"]
DOCS = [(n, n if os.path.isabs(n) else os.path.join(HERE, n))
        for n in (os.environ.get("DOCS", "").split(",") or _DEFAULT)
        if n.strip()] or [(n, os.path.join(HERE, n)) for n in _DEFAULT]


def probe(name: str, path: str) -> None:
    if not os.path.exists(path):
        print(f"\n{name}: NOT FOUND at {path}")
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = G11.normalize_text(raw)

    headers = detect_headers(text)
    cov = coverage_of(headers, text)
    usable = headers_are_usable(headers, text)

    print(f"\n{'=' * 74}")
    print(f"{name}")
    print(f"{'=' * 74}")
    print(f"  raw          {len(raw):,} chars, "
          f"{raw.count(chr(10)):,} real newlines, "
          f"{raw.count(chr(92) + 'n'):,} literal backslash-n")
    print(f"  normalized   {len(text):,} chars, "
          f"{text.count(chr(10)):,} real newlines, {len(text.split()):,} words")
    print(f"  '## ' as a plain substring:      {text.count('## '):,}")
    print(f"  '^##' line-anchored (re.M):      "
          f"{len(re.findall(r'^##', text, re.M)):,}   "
          f"<- what a line-anchored regex would find")
    print(f"  detect_headers() returned:       {len(headers):,}")
    if not headers:
        print(f"  coverage                         0.000")
        print(f"  headers_are_usable()             {usable}   -> chunking_mode "
              f"= {'headers' if usable else 'words'}")
        return

    print(f"  first header at                  {headers[0]['start']:,} "
          f"({headers[0]['start'] / len(text):.1%} into the file)")
    print(f"  last header ends at              {headers[-1]['end']:,} "
          f"({headers[-1]['end'] / len(text):.1%})")
    print(f"  coverage                         {cov:.3f}")
    print(f"  headers_are_usable()             {usable}   -> chunking_mode "
          f"= {'headers' if usable else 'words'}")

    sizes = sorted(h["end"] - h["start"] for h in headers)
    words = sorted(len(text[h["start"]:h["end"]].split()) for h in headers)
    print(f"\n  section size, chars:  min {sizes[0]:,}  "
          f"median {sizes[len(sizes) // 2]:,}  "
          f"p90 {sizes[int(len(sizes) * 0.9)]:,}  max {sizes[-1]:,}")
    print(f"  section size, words:  min {words[0]:,}  "
          f"median {words[len(words) // 2]:,}  "
          f"p90 {words[int(len(words) * 0.9)]:,}  max {words[-1]:,}")
    over = sum(1 for w in words if w > 3 * G11.WORDS_PER_READER)
    print(f"  sections over 3x WORDS_PER_READER ({3 * G11.WORDS_PER_READER:,} "
          f"words): {over}")

    counts = collections.Counter(h["title"] for h in headers)
    repeated = [(t, n) for t, n in counts.most_common() if n > 1]
    unique = [t for t, n in counts.items() if n == 1]
    print(f"\n  distinct titles:  {len(counts):,}   "
          f"repeated: {len(repeated):,}   appearing once: {len(unique):,}")
    print(f"  the 12 commonest titles (these are the RECURRING SUB-HEADERS):")
    for t, n in counts.most_common(12):
        print(f"     {n:>4}x  {t[:64]!r}")
    print(f"  12 titles appearing exactly once (candidate ENTITY names):")
    for t in unique[:12]:
        print(f"           {t[:64]!r}")


if __name__ == "__main__":
    for name, path in DOCS:
        probe(name, path)


# --------------------------------------------------------------- STEP 4b RULE
#
# Which header titles name an ENTITY the document profiles, and which are
# structural furniture. Shown here standalone, with its hit/miss list, before
# it is wired to anything.

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


def looks_like_entity_header(title: str, counts: collections.Counter) -> bool:
    """
    Six tests, all of them about the SHAPE of the title, none about parks.

    A title that fails any one of them is structural furniture, not an entity.
    """
    if not (4 <= len(title) <= 70):
        return False
    if counts[title] > 1:
        return False                       # recurring: it is the book's scaffolding
    if title.rstrip().endswith(("!", "?", "...", "…", ":")):
        return False                       # 'Do this!', 'Stay here...'
    words = _title_words(title)
    if len(words) < 2:
        return False
    if words[0].isdigit() or re.match(r"^\d", title.strip()):
        return False                       # '01 Val di Rose', '03 Estany Llong'
    proper = [w for w in words if re.match(r"^[A-Z][a-zÀ-ɏ'’-]", w)]
    if len(proper) < 2:
        return False                       # needs real title case, not ALL CAPS
    if all(w.lower() in _GENERIC_TITLE_WORD or len(w) < 3 for w in words):
        return False
    return True


# How many other candidates must share a trailing word before that word is
# accepted as the document's own naming convention for its entities.
ENTITY_SUFFIX_MIN = int(os.environ.get("ENTITY_SUFFIX_MIN", "3"))


def entity_headers(headers: list[dict]) -> list[str]:
    """
    The subset of header titles that name an entity the document profiles.

    Two stages. The shape test above throws out the furniture; then the
    survivors must agree on a NAMING CONVENTION -- at least ENTITY_SUFFIX_MIN
    of them ending in the same word. A book of profiles names its subjects
    alike ('... National Park', '... Museum', '... Ltd'); a hotel listed inside
    one profile does not join that pattern, and neither does a one-off proper
    noun in running text. Where no convention emerges, nothing is seeded --
    which is the safe answer for a document that simply does not work this way.
    """
    counts = collections.Counter(h["title"] for h in headers)
    shaped = [h["title"] for h in headers
              if looks_like_entity_header(h["title"], counts)]
    if not shaped:
        return []
    tails = collections.Counter(_title_words(t)[-1].lower() for t in shaped
                                if _title_words(t))
    convention = {w for w, n in tails.items() if n >= ENTITY_SUFFIX_MIN}
    if not convention:
        return []
    kept, seen = [], set()
    for t in shaped:
        words = _title_words(t)
        if words and words[-1].lower() in convention and t not in seen:
            seen.add(t)
            kept.append(t)
    return kept


def probe_entities(name: str, path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        text = G11.normalize_text(f.read())
    headers = detect_headers(text)
    if not headers_are_usable(headers, text):
        print(f"\n{name}: not header-mode; entity seeding never runs.")
        return
    cut = G11.find_back_matter(text)
    body = [h for h in headers if cut is None or h["start"] < cut]
    counts = collections.Counter(h["title"] for h in body)
    shaped = [t for t in dict.fromkeys(h["title"] for h in body)
              if looks_like_entity_header(t, counts)]
    kept = entity_headers(body)
    print(f"\n{'=' * 74}\n{name}: entity-header classification\n{'=' * 74}")
    print(f"  header sections in the body:      {len(body):,}")
    print(f"  distinct titles:                  {len(counts):,}")
    print(f"  pass the SHAPE test:              {len(shaped):,}")
    print(f"  pass SHAPE + naming convention:   {len(set(kept)):,}")
    tails = collections.Counter(_title_words(t)[-1].lower() for t in shaped
                                if _title_words(t))
    print(f"  naming conventions found (>= {ENTITY_SUFFIX_MIN}): "
          f"{ {w: n for w, n in tails.most_common(6) if n >= ENTITY_SUFFIX_MIN} }")
    print(f"\n  KEPT ({len(set(kept))}):")
    for t in dict.fromkeys(kept):
        print(f"     + {t}")
    dropped = [t for t in shaped if t not in set(kept)]
    print(f"\n  DROPPED by the convention test ({len(dropped)}), first 20:")
    for t in dropped[:20]:
        print(f"     - {t}")
    furniture = [t for t in counts if not looks_like_entity_header(t, counts)]
    print(f"\n  DROPPED by the shape test ({len(furniture)}), the 20 commonest:")
    for t, n in collections.Counter(
            {t: counts[t] for t in furniture}).most_common(20):
        print(f"     - {n:>3}x  {t[:56]!r}")
