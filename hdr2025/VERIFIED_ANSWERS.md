# Locally verified answers — HDR 2025

The question set ships **no `key_points` and no `forbidden`**, so there is no
local scoring. This file is the substitute: every answer below was derived in
Python from `hdr2025.txt` with no LLM involved, and each was cross-checked a
second way. After a run, diff `submission_h3.json` against these ten. If the
pipeline disagrees with a line here, the pipeline is wrong.

Reproduce all of it with:

```bash
python -c "import hdr_compile as C; t=open('hdr2025.txt').read(); print(C.summarize_table1(C.parse_annex_table1(t)))"
```

## Aggregation

**h01** — Statistical Annex Table 1, ranked entries by HDI group:

| Group | Entries |
|---|---|
| Very high human development | 74 |
| High human development | 50 |
| Medium human development | 43 |
| Low human development | 26 |
| **Total ranked entries** | **193** |

*Cross-check*: the four groups sum to 193; the lowest printed rank is 193; 193
distinct country names with no duplicates; the HDI ranges per group
(0.804–0.972 / 0.703–0.799 / 0.550–0.698 / 0.388–0.544) sit exactly inside the
report's own published cutoffs (≥0.800 / 0.700–0.799 / 0.550–0.699 / <0.550)
with no leakage across boundaries. The unranked "Other countries or
territories" block (Korea DPR, Monaco, …) is excluded, as the question says
*ranked*.

**h02** — Contents: **BOXES 19**, **SPOTLIGHTS 12**, **TABLES 6** (S-prefixed
items included: 2 boxes, 1 table).

**h03** — Contents FIGURES: **67 total**.

| Section | Figures |
|---|---|
| Overview | 8 |
| Chapter 1 | 9 |
| Chapter 2 | 4 |
| **Chapter 3** | **19 ← most** |
| Chapter 4 | 6 |
| Chapter 5 | 9 |
| Chapter 6 | 12 |

*Cross-check for h02/h03*: counting the identifiers printed in the Contents and
counting the captions printed in the body (pages 1–233) are independent
methods, and they agree exactly — 67 figures, 19 boxes, 6 tables, with
**identical identifier sets** (zero in one and not the other).

## Superlative

| | Answer | Value |
|---|---|---|
| **h04** highest HDI 2023 | **Iceland** | 0.972 (rank 1) |
| **h05** highest life expectancy 2023 | **San Marino** | 85.7 years (rank 29) |
| **h06** highest GNI per capita 2023 | **Liechtenstein** | 166,812 (2021 PPP $, rank 17) |

No ties at the top in any of the three.

## Absence

All three turn on counting the **report body (pages 1–233) only**. References
run from page 234 to the end.

| | Answer | Body | References |
|---|---|---|---|
| **h07** | **the metaverse** | 0 | 1 |
| **h08** | **digital twins** | 0 | 0 |
| **h09** | **universal basic income** | 0 | 0 |

**h07 is the trap.** Counting the whole file, `metaverse` occurs once, so a
whole-document index reports it as mentioned — the wrong answer. That single
occurrence is inside the title of a cited work:

> ITU (International Telecommunication Union). 2023a. *Cyber Risks, Threats,
> and Harms in the Metaverse.* ITU Focus Group Technical Report.

The other three subjects in h07 are discussed freely in the body:
`deepfake` 9, `care technolog…` 14, `algorithmic management` 17.

For h08 the distractors are `people with disabilities` 34, `screen time` 15,
`Global Digital Compact` 11. For h09 they are `social dialogue` 31,
`audit protocol` 3, `dangerous planetary change` 2 — note that two of these are
*rare but present*, which is exactly the discrimination the question is testing.

## Needle

**h19** — PDF ISBN **9789211542639** (page 2 copyright block; print ISBN is
9789211576092, online ISSN 2412-3129 — do not confuse them).

**h21** — **12 orders of magnitude**. Confirmed by a body sentence, not only by
the Figure 1.6 caption: *"…driven in part by the steady decline in the cost of
computing, which fell by 12 orders of magnitude…"*, and the caption of Figure
1.6 agrees — *"The cost of computing declined by 12 orders of magnitude in the
classical programming age"*.

## Not verified here

h10–h18 and h20 depend on prose and are left to the readers. Two notes from the
compiler that bear on them:

- **h13/h14** cite figures the index resolves cleanly — Figure 1.8 *"The lower
  the level of skill and experience, the more workers benefit from artificial
  intelligence (AI)"* (p46), Figure 6.2 *"Men and people with greater levels of
  education report higher use of artificial intelligence (AI) for work"* (p180),
  Figure 5.5 *"The majority of today's large-scale artificial intelligence
  models are developed by organizations based in the United States…"* (p164).
- **h20** (how the cover and chapter images were made with AI, and what the
  designer did after prompting) is pure prose in the front matter — the readers
  holding the early slices are the only source.

## Known imperfection in the source text

Endnote and reference pages (roughly 217–225 and 279–286) are dense
multi-column citation lists whose reading order is interleaved. Neither
extraction mode handles them. They are citations, so low value for these
questions, but do not write new questions against them.
