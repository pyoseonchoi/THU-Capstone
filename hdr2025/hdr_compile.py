"""
The HDR compiler: everything about this document that Python can settle.

This replaces the national-parks compiler wholesale. Nothing here calls an LLM
and nothing here looks at a question -- it runs once at ingest and produces the
same store no matter what is asked, which is what keeps the pipeline out of
retrieval territory.

Four things get compiled:

  1. STATISTICAL ANNEX TABLE 1  193 ranked rows, each with its HDI band. Gives
     the counts (h01) and the maxima (h04-h06) exactly, with no model involved.
  2. THE CONTENTS LISTS         boxes, spotlights, tables and figures with their
     identifiers, so they can be counted per chapter (h02, h03).
  3. A BODY-ONLY COVERAGE INDEX Pages 1-233 only. The report carries an 87-page
     reference section, and a term appearing solely in a cited work's title is
     not a subject the report discusses -- "metaverse" occurs exactly once, in
     an ITU report title, and the absence question h07 turns on that.
  4. A FIGURE CAPTION INDEX     the cross-section questions cite figures by
     number ("Figure 1.8 says...", "Figure 6.2 says..."), so the captions are
     worth having addressable.
"""

import re

# References begin on page 234 and run to the end; the statistical annex sits
# inside that tail. Everything the report itself argues is at or before 233.
BODY_LAST_PAGE = 233

PAGE_MARKER = re.compile(r"^\[page (\d+)\]$", re.M)

HDI_BANDS = ("Very high human development", "High human development",
             "Medium human development", "Low human development")


# ------------------------------------------------------------- PAGES

def split_pages(text: str) -> dict[int, str]:
    """Map page number -> that page's text, using the markers we wrote at conversion."""
    pages, marks = {}, list(PAGE_MARKER.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pages[int(m.group(1))] = text[m.end():end]
    return pages


def body_text(text: str) -> str:
    """Pages 1-233: the report proper, without references or the annex."""
    pages = split_pages(text)
    return "\n".join(pages[p] for p in sorted(pages) if p <= BODY_LAST_PAGE)


# ------------------------------------------- 1. ANNEX TABLE 1

# "  62 Serbia                    0.833      76.8      15.0     11.6 e   ..."
# The rank and name are one field; every value after is whitespace-separated and
# may carry a footnote letter or be ".." for missing data.
_ROW = re.compile(r"^\s{0,4}(\d{1,3})\s+([A-Z][^\s].*?)\s{2,}(.+)$")
# A cell is '..' (missing) or a number that must contain at least one digit --
# '[\d,]+' alone happily matches a lone comma and yields '' after stripping.
_VALUE = re.compile(r"(\.\.|-?\d[\d,]*(?:\.\d+)?)\s*[a-z]{0,2}(?=\s|$)")

_TABLE1_FIELDS = ("hdi", "life_expectancy", "expected_schooling",
                  "mean_schooling", "gni_per_capita", "gni_rank_minus_hdi_rank",
                  "hdi_rank_2022")


def _numbers(tail: str) -> list:
    """Pull the numeric cells out of a row's tail, keeping '..' as None."""
    out = []
    for m in _VALUE.finditer(tail):
        raw = m.group(1)
        out.append(None if raw == ".." else float(raw.replace(",", "")))
    return out


def parse_annex_table1(text: str) -> list[dict]:
    """
    Every ranked row of Statistical Annex Table 1, tagged with its HDI band.

    The band is carried by a bare section heading ("Very high human
    development") that precedes its block; the same phrase recurs later as an
    aggregate row, which is why the heading test requires the line to hold no
    digits. Rows under "Other countries or territories" are unranked and are
    deliberately excluded -- h01 asks for ranked entries.
    """
    pages = split_pages(text)
    lines = []
    for p in sorted(pages):
        if 288 <= p <= 292:                      # table 1 and its notes
            lines.extend(pages[p].split("\n"))

    rows, band = [], None
    for line in lines:
        stripped = line.strip()
        if not any(c.isdigit() for c in stripped):
            for candidate in HDI_BANDS:
                if stripped.lower() == candidate.lower():
                    band = candidate
            if stripped.lower().startswith("other countries"):
                band = None                       # unranked tail: stop collecting
            continue
        if band is None:
            continue

        m = _ROW.match(line)
        if not m:
            continue
        rank, name, tail = int(m.group(1)), m.group(2).strip(), m.group(3)
        # The running footer is "276    HUMA N D EVELOP MENT R EP ORT 2025".
        if "EVELOP" in name and "MENT" in name:
            continue
        values = _numbers(tail)
        if not values or values[0] is None or not (0.0 < values[0] < 1.0):
            continue                              # not an HDI row

        row = {"rank": rank, "country": name, "hdi_band": band}
        row.update(dict(zip(_TABLE1_FIELDS, values)))
        rows.append(row)
    return rows


def summarize_table1(rows: list[dict]) -> dict:
    """The counts and maxima h01 and h04-h06 ask for, computed not inferred."""
    by_band = {b: [r for r in rows if r["hdi_band"] == b] for b in HDI_BANDS}

    def top(field: str) -> dict | None:
        have = [r for r in rows if r.get(field) is not None]
        if not have:
            return None
        best = max(have, key=lambda r: r[field])
        tied = [r["country"] for r in have if r[field] == best[field]]
        return {"country": best["country"], "value": best[field],
                "rank": best["rank"], "tied_with": [c for c in tied
                                                    if c != best["country"]]}

    return {
        "ranked_entries_total": len(rows),
        "ranked_entries_by_hdi_group": {b: len(v) for b, v in by_band.items()},
        "highest_hdi_2023": top("hdi"),
        "highest_life_expectancy_2023": top("life_expectancy"),
        "highest_gni_per_capita_2023": top("gni_per_capita"),
        "lowest_rank": max((r["rank"] for r in rows), default=None),
        "note": ("Parsed directly from the printed rows of Statistical Annex "
                 "Table 1. Ranked entries only; the unranked 'Other countries "
                 "or territories' block is excluded. These counts and maxima "
                 "are authoritative."),
    }


# --------------------------------------------- 2. THE CONTENTS LISTS

# An entry begins with its identifier, then whitespace, then its title:
#   "5.4      The potential for artificial intelligence audit protocols   155"
# Overview figures are lettered O.1, O.2...; spotlight items carry a leading S.
_CONTENTS_ID = re.compile(r"^(S?(?:O|\d+)(?:\.\d+)+)\s{2,}\S")

CONTENTS_PAGES = (11, 12, 13)

# The headings are set with letter-spacing, so they arrive as "F I G U R ES",
# "TABL ES", "S POTL I G H TS". Compare with the spaces removed.
CONTENTS_SECTIONS = ("BOXES", "SPOTLIGHTS", "FIGURES", "TABLES")

# Cutting the page at its gutter leaves the neighbouring column's page number
# stuck to the heading -- "62             BOXES", "S POTL I G H TS ... 1" -- so
# strip digits off both ends before comparing.
_EDGE_DIGITS = re.compile(r"^\d*|\d*$")


def _heading(line: str) -> str | None:
    squashed = _EDGE_DIGITS.sub("", line.replace(" ", "").upper())
    return squashed if squashed in CONTENTS_SECTIONS else None


def parse_contents(text: str) -> dict:
    """
    The identifiers listed under each Contents heading.

    h02 and h03 are pure counting questions over these lists, including the
    S-prefixed spotlight items, which is why the identifier pattern allows a
    leading S rather than filtering it out.

    These pages were column-split at conversion, so each column is a coherent
    stream and a heading governs every entry until the next heading.
    """
    pages = split_pages(text)
    lines = []
    for p in CONTENTS_PAGES:
        if p in pages:
            lines.extend(pages[p].split("\n"))

    sections, current = {}, None
    for line in lines:
        stripped = line.strip()
        heading = _heading(stripped)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        if not current:
            continue
        m = _CONTENTS_ID.match(stripped)
        if m and m.group(1) not in sections[current]:
            sections[current].append(m.group(1))

    def section_of(identifier: str) -> str:
        head = identifier.lstrip("S").split(".")[0]
        return "Overview" if head.upper() == "O" else f"Chapter {head}"

    summary = {}
    for name, ids in sections.items():
        per_section = {}
        for i in ids:
            per_section[section_of(i)] = per_section.get(section_of(i), 0) + 1
        ordered = dict(sorted(per_section.items(),
                              key=lambda kv: (kv[0] != "Overview", kv[0])))
        summary[name] = {
            "identifiers": ids,
            "count": len(ids),
            "count_by_section": ordered,
            "section_with_most": max(per_section, key=per_section.get)
                                 if per_section else None,
            "spotlight_items_included": sum(1 for i in ids if i.startswith("S")),
        }
    return summary


# ------------------------------------- 3. BODY-ONLY COVERAGE INDEX

def coverage(text: str, terms: list[str]) -> list[dict]:
    """
    Count each term in the body and in the reference tail, separately.

    Keeping the two apart is the whole point. A term that occurs only in the
    reference section appears in a cited work's TITLE and is not a subject the
    report discusses -- so 'mentioned in the report' must mean body_mentions>0,
    not total>0. h07 hangs on exactly this: 'metaverse' occurs once in the
    whole file, inside an ITU report title, and is the correct answer to
    "which subject is never mentioned".
    """
    pages = split_pages(text)
    body = "\n".join(pages[p] for p in sorted(pages) if p <= BODY_LAST_PAGE).lower()
    tail = "\n".join(pages[p] for p in sorted(pages) if p > BODY_LAST_PAGE).lower()

    out = []
    for term in terms:
        t = term.lower()
        b, r = body.count(t), tail.count(t)
        out.append({
            "term": term,
            "body_mentions": b,
            "reference_section_mentions": r,
            "total_mentions": b + r,
            "discussed_in_report": b > 0,
        })
    return sorted(out, key=lambda d: d["body_mentions"])


# ------------------------------------------ 4. FIGURE CAPTION INDEX

_FIGURE = re.compile(r"^\s*(?:Figure|FIGURE)\s+(S?\d+(?:\.\d+)+)\s*(.*)$")


def build_figure_index(text: str) -> list[dict]:
    """
    Figure number -> caption and page.

    The cross-section questions name figures directly ("Figure 5.5 says most
    large-scale AI models are developed by organizations in the United
    States"), so an addressable caption list turns a search into a lookup.
    """
    pages, seen = split_pages(text), {}
    for page in sorted(pages):
        lines = pages[page].split("\n")
        for i, line in enumerate(lines):
            m = _FIGURE.match(line)
            if not m:
                continue
            caption = m.group(2).strip()
            j = i + 1
            while j < len(lines) and len(caption) < 40:
                caption = (caption + " " + lines[j].strip()).strip()
                j += 1
            if m.group(1) not in seen and caption:
                seen[m.group(1)] = {"figure": m.group(1), "page": page,
                                    "caption": " ".join(caption.split())[:300]}
    return [seen[k] for k in sorted(seen)]
