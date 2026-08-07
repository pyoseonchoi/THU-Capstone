# g12 — header-aware segmentation: what was built and what it found

All numbers below are deterministic. Nothing here cost a broker call.

## Step 1–2: the gate (`detect_headers`, `headers_are_usable`)

Run standalone via `header_probe.py`, before anything was wired:

| document | `## ` substrings | headers found | coverage | mode |
|---|---|---|---|---|
| `nationalparks_europe.txt` | 1,007 | **1,006** | **1.000** | `headers` |
| `oecd2026_fullscan_test.txt` | 0 | 0 | 0.000 | `words` |
| `gem2024_5_fullscan_no_indent.txt` | 0 | 0 | 0.000 | `words` |
| `hdr2025.txt` | 0 | 0 | 0.000 | `words` |

The gate separates them across the entire range of the measure, so `min_count=5`
and `min_coverage=0.5` are not delicate. Detection is by substring, not
line-anchored: the parks file is one line of 578,358 characters with literal
backslash-n where its newlines should be.

## Step 3: sections are the ATOM, not the chunk

The spec said one chunk per section. Measured, that is wrong for this document:
1,006 sections over 92,808 words is a **median section of 53 words**, because
most headers are the sub-headings repeated inside every entry (`Toolbox` ×57,
`Do this!` ×60, `What to spot...` ×60) rather than one per park. One chunk per
section is **1,006 readers where the word chunker makes 15** — 65× the model
calls, against the hard constraint, and every reader far below the ~6,000 words
that is the measured line between enumerating and summarising.

So `chunk_by_headers` packs whole sections up to `WORDS_PER_READER`. Same reader
count, same reader size, boundaries that are the document's own:

| chunker | slices | cuts landing inside a section |
|---|---|---|
| g11 `chunk_lines` | 16 | **15 of 15** |
| g12 `chunk_by_headers` | 15 | **1 of 14** |

`HEADER_PACK_SLACK = 0.05` exists because packing atoms wastes bin capacity that
free cutting does not; without it the parks book needs 16 slices where g11 needs
15, i.e. the change would have ADDED calls. With it: **−3 calls** over 21
questions.

## Step 4a: `section_title` into the arc reader

`ARC_PROMPT_HEADERS` hands the reader its section headings as printed instead of
asking it to infer the `SECTION:` line. Separate template, so the words-mode
string stays byte-identical.

## Step 4b: the entity classification rule — and why it declines here

Six shape tests (`looks_like_entity_header`), then the survivors must agree on a
naming convention. Measured on the parks book: 178 titles pass the shape test,
and their trailing words are **park 18, hotel 15, house 4, spa 3**.

The book gives a header to only **18 of the 60 parks** it profiles, but to nearly
every hotel inside a `Stay here...` block. `park` takes 45% of the survivors and
beats `hotel` by 1.2×; the thresholds are 60% and 2.0×, so **nothing is seeded**.
That is the right answer — seeding would have put `Fossli Hotel`, `Radisson Blu
Hotel` and 19 other non-entities in front of every counting reader.

A hierarchy discriminator was tried and rejected: median sections between members
is 34 for `park` and 44 for `hotel`, which does not separate them, because only
30% of parks carry a header at all.

## The finding: the bottleneck is neither boundaries nor extraction

Replaying `submission_g11_parks.json` through `measure_recall.py`:

```
d02 highest point >= 3000 m (8 parks)   extracted 8/8 = 100%
d01 parks in Spain (6 parks)            extracted 6/6 = 100%
d03 parks in Croatia (3 parks)          extracted 3/3 = 100%
```

Extraction recall is already perfect, and all 8 target parks already had name and
figure in the same slice under the old chunker. The loss is entirely **between
records and answer**, in `verify_record`:

```
d02: 82 records in
  after provenance: 37    rejected: value_not_near_entity 44, entity_not_found 1
  MATCHING >= 3000: 3
```

Five of the eight correct parks are discarded there. The park's name is a heading
and its `Park in numbers` box sits further down:

| park | distance name → figure | `PROVENANCE_WINDOW` |
|---|---|---|
| Hohe Tauern | 2,346 | 1,200 |
| Sierra Nevada | 2,380 | 1,200 |
| Ordesa | 2,404 | 1,200 |
| Aigüestortes | 2,429 | 1,200 |
| Pyrenees | 2,922 | 1,200 |

## `provwindow` — the fix the headers make principled (default OFF)

Raising the constant to 3,000 would be fitting the code to one document.
`entry_period()` instead measures the median gap between consecutive occurrences
of the document's commonest repeated heading — the same evidence `count_entries`
already trusts — and takes half of it. On the parks book: period 9,874 →
window **4,937**.

Same records, same code, one flag:

| | window 1,200 | window 4,937 |
|---|---|---|
| d02 matching | 3 of 19 entities | **8 of 47** |
| d02 named | Écrins, Etna, Swiss | **Écrins 4102, Hohe Tauern 3798, Sierra Nevada 3479, Etna 3350, Ordesa 3348, Pyrenees 3298, Swiss 3174, Aigüestortes 3033** |

That is the answer key's 8 exactly, and it trips none of d02's forbidden claims
(no Jotunheimen 2469, no Triglav 2864, no Valbona 2694).

Default OFF because it moves a number every counting question depends on, and the
standing rule is one change per scored run.

## Regression proof (`header_equivalence.py`)

Not a re-run — a byte comparison of every prompt g11 and g12 build. Since the
synthesis prompt embeds the computed block, identical prompts prove the tally
arithmetic is identical too.

| case | slices | call delta | prompts identical |
|---|---|---|---|
| OECD, all four fixes on | 28 = 28 | **+0** | **21/21** |
| GEM, all four fixes on | 44 = 44 | **+0** | **21/21** |
| HDR, all four fixes on | 35 = 35 | **+0** | **21/21** |
| Parks, `G12_FIXES=` | 15 = 15 | +0 | 21/21 |
| Parks, fixes on | 15 = 15 | **−3** | 3/21 (must differ) |

## Still open — not g12's scope

`d01`'s "how many are in Spain" is a separate defect. `tally()` only reaches
`categorical_tally()` when NO record is numeric; d01's records include areas, so
the country records are never counted. Even when forced, `categorical_tally`
picks `highest point` as its dominant non-numeric attribute and groups on summit
names. That is the next thing to fix, and it is unrelated to headers.
