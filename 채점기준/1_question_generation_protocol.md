# Protocol 1 — Question Generation

**Purpose:** Given a large source document, produce a standardized question set as
JSON, in the format used by the Long-Context Challenge. Hand this file to any AI
chatbot together with the source document; the procedure is written to give
consistent results regardless of which chatbot runs it.

**Input:** one plain-text source document.
**Output:** one `questions.json` file (schema below).

> **Bias note.** If the *same* model that generates questions is later used to
> answer or score them, questions drift toward what that model finds easy. This
> toolkit is for practice and internal comparison, not for predicting the real
> competition, where questions come from the organizers. Do not read scores
> produced from self-generated questions as competition forecasts.

---

## What the set must contain

- **21 questions total: 7 categories × 3 questions each.**
- Categories, in fixed order: `aggregation`, `superlative`, `absence`,
  `contradiction`, `cross_section`, `global_synthesis`, `needle`.
- IDs are sequential, zero-padded: `q01` … `q21`. Assign them in category order
  (q01–q03 aggregation, q04–q06 superlative, …).
- Six of the seven categories are chosen because retrieval alone cannot answer
  them. Each question must genuinely require reading across the whole document,
  **except** the `needle` control category.

## Category definitions and construction rules

For every question, you must be able to point to the exact text that makes the
answer true. **Do not invent facts.** If the document does not support a clean
question in some category, search harder before settling — see per-category
notes.

- **aggregation** — counting or totalling something over the *entire* document
  (distinct entities, events, occurrences). Construction: pick an attribute that
  recurs across many sections (e.g. a stat that appears in every entry's data
  box). The answer must be a count that requires visiting every section, plus the
  named items. Verify the count by scanning the whole document, not a sample.

- **superlative** — a maximum/minimum/earliest/most-frequent taken over the whole
  corpus. Construction: choose a comparable quantity present in many sections;
  the answer is the single extreme value and the entity holding it. **Prefer
  questions where the naive answer (the biggest value in one prominent section)
  differs from the true global extreme** — these are the discriminating ones.

- **absence** — what the document never substantively discusses. Construction:
  offer three or four named subjects, three of which *are* covered and one of
  which is *not*. Verify absence by searching the whole text for the term and its
  synonyms; a term appearing only inside a citation title or reference list
  counts as "not substantively discussed." Verify the three present ones actually
  appear in substance.

- **contradiction** — two places where the document states incompatible things
  about the same factual matter. Construction: find a real internal
  inconsistency (a figure, rank, or claim stated one way in one section and
  another way elsewhere). These are the hardest to find and must be *genuine* —
  do not plant or manufacture them. If the document is a tightly edited single
  work, real contradictions may be rare; spend real effort here and, if none can
  be verified, record that in `notes` rather than inventing one.

- **cross_section** — one fact combined with another that sits far away, where the
  document never states the link. Construction: pick two facts from distant
  sections (e.g. a named entity in a narrative passage + that same entity's row
  in a data table) and ask a question whose answer requires joining them.

- **global_synthesis** — a theme and how it develops across the document: what
  changes, what stays constant. Construction: identify an argument or framing
  that shifts between early and late sections; the answer describes the
  trajectory, referencing where the change happens.

- **needle (control)** — one obscure fact stated in exactly one place, phrased
  much like the question. Construction: pick a single specific detail (a date, a
  name, an ISBN, an origin story). This is the control; a retrieval pipeline
  should score near 100% here.

## Question-writing style

- Each question is self-contained and answerable from the document alone.
- Ask for a committed answer, not a list of possibilities.
- Where a count or named list is expected, say so explicitly ("how many … and
  name them").
- Keep the wording neutral; do not leak the answer.

## Output schema

`questions.json` is a JSON array. Each element:

```json
{
  "category": "aggregation",
  "id": "q01",
  "question": "How many national parks does the book profile in total, and how many of them are in Spain? Name the Spanish ones."
}
```

- `category` — one of the seven fixed category names.
- `id` — `q01`…`q21`, sequential in category order.
- `question` — the question text, a single string.

Nothing else. The file must be UTF-8.

## Procedure (run this each time)

1. Read the whole document. Note its structure (sections, data boxes, tables,
   narrative arc, front/back matter).
2. For each category in order, draft 3 questions using the rules above.
3. For every drafted question, locate and confirm the supporting text. If you
   cannot confirm it, replace the question.
4. Give special, explicit effort to `contradiction`: search for the same
   attribute stated in two places with different values. Only include verified
   ones.
5. Emit `questions.json` in the schema above, IDs assigned in category order.
6. In your reply (not in the file), list any category where you could not find a
   fully verified question, so the human can decide how to proceed.
