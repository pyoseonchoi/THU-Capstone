# Protocol 3 — Scoring

**Purpose:** Given a source document, a frozen `answer_key.yaml`, and a model's
submission JSON, produce per-question scores and an overall quality score, using
the Long-Context Challenge formula. Hand this file to any AI chatbot together
with those three inputs to get a consistent, baseline score every time.

**Inputs:**
1. the source document (for verification),
2. `answer_key.yaml` (frozen, from Protocol 2),
3. a submission JSON (the model's answers).

**Output:** a scoring report (per-question + overall), format below.

> If no `answer_key.yaml` is supplied for this document, stop and run Protocol 2
> first to build one, then freeze it and score against it. Never score against a
> key invented on the spot during scoring — that breaks reproducibility.

---

## The formula (apply exactly)

For each question:

```
question_score = max(0, covered / total_key_points − 0.25 × forbidden_hits)
```

- `total_key_points` — number of key points for that question in the answer key.
- `covered` — how many of those key points are present in the submitted answer.
- `forbidden_hits` — how many forbidden claims the submitted answer states.

Overall:

```
quality = mean of question_score over ALL questions in the answer key
```

- The mean is over **every** question in the key, not only the answered ones.
- A question with no submitted answer (or a blank/`error` answer) scores **0**.
  Missing costs exactly as much as wrong.

## Judging rules (apply to every question)

- **Verify against the source.** Before marking a key point present, confirm the
  submitted answer's claim is actually correct per the source document — a
  fluent-but-wrong statement does not cover a key point. Before counting a
  forbidden hit, confirm the answer really makes that forbidden claim. This
  verification step is mandatory for every question.
- **Present/absent is binary per key point.** Partial phrasing that clearly
  states the fact counts as present; a vague gesture at it does not.
- **Hedging does not count as covering.** "It might be 5, 6, or 7" does not cover
  the key point "states 7", and listing alternatives can additionally trip a
  forbidden claim. Score hedged answers on what they commit to, which is often
  nothing.
- **Padding is neutral.** Extra material neither adds nor subtracts, unless it
  states something on the forbidden list — then it counts as a forbidden hit.
- **Clamp at zero.** `question_score` is never negative; `max(0, …)` applies per
  question before averaging.
- **One answer per id.** If a submission repeats an id, score the first
  occurrence and note the duplicate.
- **Unknown ids are ignored** (with a note); they do not earn or cost points.

## Procedure (run this each time)

1. Load the answer key. Establish the full list of question ids and each
   question's `total_key_points`.
2. Load the submission. Map answers to ids. Note any missing ids (they will score
   0), blanks/errors (0), duplicates, and unknown ids.
3. For each question id in the answer key:
   a. Take the submitted answer (or 0 if missing/blank/error).
   b. For each key point, verify against the source and mark present/absent.
      Count `covered`.
   c. For each forbidden claim, verify against the source and the answer; count
      `forbidden_hits`.
   d. Compute `question_score = max(0, covered/total_key_points − 0.25 ×
      forbidden_hits)`.
4. Compute `quality` as the mean over all questions in the key.
5. Emit the report below.

## Report format

Produce, in this order:

1. A per-question table:

   | id | category | covered/total | forbidden_hits | score | note |
   |----|----------|---------------|----------------|-------|------|

   `note` briefly states why points were lost (e.g. "hedged on the count",
   "named second-highest — forbidden hit", "missing answer").

2. A category breakdown: mean score per category (7 rows).

3. The overall `quality` score, as a decimal, and the raw sum (e.g.
   "quality = 0.80, sum 16.8 / 21").

4. A short "biggest point losses" list: the 3–5 questions where the most points
   were lost, each with the specific fix.

## Reproducibility checklist (state these in the report)

- Answer-key file name/version used.
- Number of questions scored, missing, duplicated, unknown.
- Confirmation that every key point and forbidden hit was verified against the
  source document.
