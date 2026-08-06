# Protocol 2 — Answer Key Generation

**Purpose:** Given a source document and a `questions.json`, produce a **frozen
answer key** as YAML — the checklist of atomic key points and forbidden claims
that scoring (Protocol 3) runs against. Freezing the key once is what makes
scores reproducible: every later scoring run, on any chatbot, judges against the
*same* key, so the same submission always gets the same score.

**Input:** source document + `questions.json`.
**Output:** one `answer_key.yaml` (schema below).

> Run this **once** per document, then save the result and reuse it forever. Do
> not regenerate it per scoring run — that would reintroduce drift.

---

## What a key entry contains

For each question, the key holds:

- `key_points` — a list of **atomic** facts a correct answer must state. Atomic
  means one checkable claim per item (a single number, name, or relationship),
  phrased so a judge can mark it plainly present or absent. Prefer 3–6 key points
  per question; more for aggregation/cross_section where several named items are
  expected.
- `forbidden` — a list of specific wrong claims that should be penalized,
  **including the answer a naive top-k retrieval pipeline would give**. Each is a
  concrete, checkable statement (e.g. "names X as the largest", "gives the count
  as 5 or fewer").

## Rules for writing key points

- **Verify every key point against the source document before writing it.** If
  you cannot point to the exact supporting text, do not include it. This is
  mandatory and non-negotiable — an unverified key point corrupts every future
  score.
- Make each key point independently checkable. Split compound facts ("Iceland,
  and its HDI is 0.972") into two items.
- State the fact, not the reasoning. "Names Iceland as highest" not "the reader
  should compare all countries."
- For counts, the key point is the exact number *and* (separately) the named
  items expected.
- Phrase key points so that a confident correct answer matches them and a hedge
  ("maybe 5, 6, or 7") does not.

## Rules for writing forbidden claims

- Always include the **naive-retrieval trap**: the plausible-but-wrong answer
  that comes from seeing only part of the document (e.g. the largest value in one
  prominent section, mistaken for the global maximum; an undercount from seeing
  only some sections; concluding a topic is present because related passages
  exist).
- Include common inversions and off-by-one errors specific to the document
  (naming the second-highest as highest, placing an entity in the wrong group).
- Each forbidden item must be a concrete claim, checkable as present/absent in an
  answer.

## Output schema

`answer_key.yaml` is a YAML list, one entry per question:

```yaml
- id: q01
  category: aggregation
  key_points:
    - "First atomic fact the answer must state"
    - "Second atomic fact"
  forbidden:
    - "A specific wrong claim, including the naive-retrieval answer"
    - "Another wrong claim"
```

- `id` must match a question id from `questions.json` exactly.
- `category` mirrors the question's category.
- Every question in `questions.json` must have exactly one key entry.

## Procedure (run this once per document)

1. Read the whole document.
2. For each question in `questions.json`:
   a. Determine the correct answer by reading the relevant text (for
      aggregation/superlative/absence, scan the *entire* document, not a sample).
   b. Write the atomic `key_points`, verifying each against the text.
   c. Write `forbidden`, always including the naive-retrieval trap.
3. Emit `answer_key.yaml` covering all 21 questions.
4. Freeze it: save it unchanged and reuse it for all future scoring of this
   document. If you ever revise a key point, treat it as a new key version and
   re-score any past submissions you want to compare.
