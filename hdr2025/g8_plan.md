# g8 — fixing the unrouted categories

A rewrite of `unrouted_categories_fix.md`, corrected against the code as it
actually stands in `pipeline_g7.py` and against the g7 score, which arrived
after that document was written.

The central argument of the source document is right and I am keeping it:
**the three failing categories are exactly the three with no ingest-time state,
and the fix is to build that state.** What follows changes the order of the
work, corrects five things that will not run as written, and adds one finding
from the g7 scoring report that reverses part of the plan.

---

## 1. What changed: arc mode was scored, and it lost

The source document lists arc mode as priority 2, "already written; apply and
test against q18 as the control." That test has now run.

| | g6 | g7 arc | Δ |
|---|---|---|---|
| q16 | 0.286 | **0.429** | +0.143 |
| q17 | 0.143 | **0.000** | −0.143 |
| q18 (control) | 0.833 | **0.667** | −0.166 |
| **subset mean** | **0.421** | **0.365** | **−0.055** |

Arc mode is a **net regression** and must not ship as written. The retention
half worked — all 28 readers kept in document order, and q16 gained the
per-chapter attribution it was previously marked down for missing. The
synthesis half is what lost the points.

So the mode table in the source document is stale in two places:
`global_synthesis` already routes to `arc` in g7, and its 0.421 is g6's
`prose` number, not arc's 0.365.

---

## 2. Five things that will not run as written

These are not quibbles; each one is a blocker.

### 2.1 `parse_records()` does not parse the document — it parses reader replies

This is the important one, because the whole "close to free" cost argument for
the claim ledger rests on it.

```python
_RECORD  = re.compile(r"^\s*RECORD\s*\|(.+?)\|(.+?)\|(.+?)\|(.*)$", re.I | re.M)
_EXTREME = re.compile(r"^\s*(MAXIMUM|MINIMUM)\s*\|(.+?)\|(.+?)\|(.*)$", re.I | re.M)
```

`parse_records(text, source)` scans for `RECORD | entity | attribute | value |
where` lines. Those lines exist only because a model was told to emit them.
Calling it on raw chunk text, as the source document's `build_claim_ledger`
does, returns an **empty list on every document**.

The claim ledger therefore requires a real ingest pass — one model call per
chunk — before any Python runs. That does not sink the idea (see §4 for why
it is still cheap), but the plan has to say so.

### 2.2 `chunk_lines` returns dicts, and that part is right

```python
return [{"n": i + 1, "of": len(out), "text": t} for i, t in enumerate(out)]
```

`c["n"]`, `c["of"]`, `c["text"]` are all real. Keep that code as written.

### 2.3 `r["value"]` is a float or `None`, not a printable string

`_make_record` stores the parsed number in `value` and the original string in
`raw_value`:

```python
{"entity": ..., "attribute": ..., "value": 1151.0, "unit": "sq miles",
 "unit_family": "sq km", "normalized": 2980.5, "raw_value": "1151 sq miles",
 "where": ..., "source": ..., "kind": "record"}
```

Building `conflicting_values` from `r["value"]` prints `1151.0` and throws away
the unit — which is precisely the confusion the `unit`/`unit_family` split
exists to prevent. Use `raw_value` for display and `normalized` for comparison.

Also note `normalized` is `None` for records whose value carries no digits
("party to the convention"). Those still contradict each other and still need
comparing, just as strings.

### 2.4 There is no `call_llm_json`

The only entry point is `call_llm(prompt) -> str`. The retry-and-repair layer
the source document refers to does not exist in this pipeline. If we want
structured output at the API level, that is new code, not a flag on existing
code.

### 2.5 The model is a module global, not a per-call argument

```python
MODEL = os.environ.get("MODEL", "mistral-small-3-2")
...
response = get_client().chat.completions.create(model=MODEL, messages=messages, ...)
```

Splitting extraction onto `ministral-3b-2512` and adjudication onto
`mistral-small-3-2` — a genuinely good idea, since extraction is where the
token volume lives — requires threading a `model=` parameter through
`call_llm`, `_run_readers`, and every job tuple. That is a mechanical change,
but it is a change, and it should be made and tested on its own rather than
folded into a category fix.

---

## 3. The finding the source document does not have

From the g7 scoring report, on why q17 scored **zero** despite being a
detailed, on-topic, factually sound answer:

> Every one of this key's 7 points for q17 bundles 2–4 distinct named facts,
> and a bundle only counts when all its named facts are present. […] this
> submission is thorough but organizes around a different narrative frame than
> the one the frozen key was built around, so it consistently lands adjacent to
> each point rather than on it.

**Coherent synthesis is a scoring liability.** A good essay subordinates facts
to a thesis and drops the ones that do not fit; the key is a checklist of named
facts and does not care about the thesis. q17 wrote a strong argument about
"structural power imbalances" and never enumerated `housing / childcare /
portability / access to services` as a set, never named `business entry`, never
named `knowledge diffusion`. Zero.

Given `question score = max(0, covered/total − 0.25 × forbidden)`, where padding
is free and omission is fatal, the instruction to the synthesis model should be
the opposite of what arc mode currently says. **Enumerate; do not argue.**

One corroborating detail worth noting: the fact q17's key wanted about *who*
moves is present in the submission — in **q18's** answer. The readers found it.
The synthesis filed it under the wrong question. That is an assembly failure,
not a retrieval failure, which means the material for a much better score is
already being extracted and thrown away.

This applies to contradiction and cross-section too, and it is free to apply.

---

## 4. The cost argument, corrected

The source document calls the ledger "close to free" for the wrong reason
(no new reading pass). Here is the right reason.

Current architecture: **every question re-reads every chunk.** On OECD that is
28 chunks × 21 questions ≈ **588 reader calls**, plus synthesis and gap-fill.

An ingest pass is **28 calls, once**. That is roughly **+5%** on a run — and on
`ministral-3b-2512` rather than `mistral-small-3-2`, less than that in spend.

So the conclusion survives, and there is a much larger prize behind it that I am
explicitly **not** proposing for g8: if per-chunk state were built once, most
question modes could stop re-reading the document entirely, and the run would
get roughly 20× cheaper. That is the right architecture and the wrong week for
it. Note it for after the deadline.

---

## 5. The g8 build, in priority order

Each step is scored before the next begins, with `absence` (1.00) and `needle`
(1.00) as standing controls. That ordering is the source document's own advice
and it is correct.

### Step 0 — repair mojibake in `normalize_text` (no model calls)

`oecd2026_fullscan_test.txt` is double-encoded. This is currently propagating
into answers:

```
PÃ´les de compÃ©titivitÃ©   →   Pôles de compétitivité
TÃ¼rkiye                    →   Türkiye
Î²- and Ï-convergence       →   β- and σ-convergence
Chapters 2â3               →   Chapters 2–3
```

Every accented proper noun in that document is mangled, and proper nouns are
exactly what key points ask us to name. If the key says *Türkiye* and the answer
says *TÃ¼rkiye*, that is a key point earned and lost to an encoding.

```python
# A latin-1-decoded UTF-8 lead byte followed by a continuation byte: the
# second character always lands in U+0080-U+00BF ("Ã´", "Ã¼", "Î²").
_MOJIBAKE = re.compile(r"[ÃÂÎÏ][-¿]")

def repair_mojibake(text: str) -> str:
    """
    Undo UTF-8 bytes that were decoded as latin-1 ("PÃ´les" -> "Pôles").

    Only fires when the telltale sequences are dense enough to be systematic,
    and silently keeps the original if the round-trip fails -- a document that
    genuinely contains "Ã" as a character must not be corrupted by the repair.
    """
    if len(_MOJIBAKE.findall(text)) < 20:
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    _log(f"preprocess    repaired {len(_MOJIBAKE.findall(text)):,} mojibake sequences")
    return repaired
```

Called from `normalize_text` before the existing zero-width strip. Deterministic,
verifiable offline at zero cost, and a no-op on HDR and parks — which are clean,
and which are therefore also its regression test.

### Step 1 — rewrite arc synthesis as enumeration, not argument

Undoes a measured regression and needs no new architecture. Keep patch Edits 1
and 3 (mode routing, full reader retention — those earned their keep). Replace
Edits 2 and 4.

The new `ARC_SYNTHESIS_PROMPT` should instruct, in this order:

1. **No thesis.** Do not open with a framing sentence or a governing argument.
2. Walk the document's **top-level** divisions in order. Do not treat a numbered
   subsection as its own section. *(This is the specific q18 fix: naming ~15
   subsections crowded out two of the six top-level stages the key wanted.)*
3. One block per division: what it claims, then **every** named entity, country,
   institution and figure it carries.
4. A `CHANGE:` block and a `CONSTANT:` block. The constant must be **quoted in
   the document's own recurring wording**, not paraphrased into an abstraction.
   *(q16 and q17 both invented "structural power imbalances" against a key that
   says "anticipate structural change / fair transitions." Asking a model to
   abstract a constant gets the model's abstraction.)*
5. A final `ALSO:` dump of extracted facts that did not fit above. Padding is
   free; omission is fatal.

Control: q18. Ship only if the subset mean clears **0.421**, not merely 0.365.

### Step 2 — claim ledger for contradiction

The category is at 0.00 and the g6 failures were *invented* number pairs
(100/101, 2 289/2 288 for a real 408/409), which is exactly what a model does
when nothing ever compared two values for the same attribute.

**2a. Ingest pass.** One call per chunk, emitting the existing RECORD format so
`parse_records` reads it unchanged:

```
RECORD | entity | attribute | value | where
```

**2b. Deterministic candidate filter.** Python, not a model:

```python
def build_claim_ledger(records: list[dict], chunk_count: int) -> list[dict]:
    """
    Group ingested records by (entity, attribute). Any key holding more than one
    distinct value is a contradiction candidate.

    This is the cheap filter that makes the problem tractable: it proposes O(n)
    candidate pairs instead of the all-pairs comparison, which for global
    consistency checking is provably exponential.
    """
    ledger = {}
    for r in records:
        key = (r["entity"].lower().strip(), r["attribute"].lower().strip())
        ledger.setdefault(key, []).append(r)

    candidates = []
    for (entity, attribute), rs in ledger.items():
        # Compare on `normalized` where a unit was parsed, so 'sq miles' and
        # 'sq km' do not read as a disagreement; fall back to the raw string for
        # records whose value carries no digits at all.
        def compare_key(r):
            return round(r["normalized"], 6) if r["normalized"] is not None \
                else r["raw_value"].lower().strip()

        if len({compare_key(r) for r in rs}) < 2:
            continue

        chunks = {r["source"] for r in rs}
        candidates.append({
            "entity": entity, "attribute": attribute,
            "chunks": sorted(chunks),
            # A disagreement inside one slice is usually a parsing artifact. The
            # real ones are far apart -- which is exactly why both humans and
            # retrieval miss them.
            "separated": len(chunks) > 1,
            "values": [{"value": r["raw_value"], "where": r["where"],
                        "source": r["source"]} for r in rs],
        })

    candidates.sort(key=lambda c: (not c["separated"], -len(c["chunks"])))
    return candidates
```

**2c. One adjudication call** on the shortlist, on `mistral-small-3-2`, asking
whether both statements can simultaneously be true and rejecting differences
that are merely different years, units, or subsets. Survivors go into the
`computed` block, which `SYNTHESIS_PROMPT` already treats as authoritative.

**2d. Extend the existing no-invention rule** from names to numeric pairs: the
synthesis model may not state a conflicting pair that is not in the ledger.
This is the direct fix for the g6 fabrications.

### Step 3 — entity lookup for cross-section

Most work, least certain, and it carries an open question (§7) that must be
answered before it is worth building. Deliberately last.

The reduced version — look up every chunk containing the question's named
entities and feed those chunks in full — only works **if the question names the
entities**. The source document's own q14/q15 example suggests it does not
("your pipeline named the United States and Sweden because it never saw
Canada's and Italy's mentions side by side" describes a question asking *which*
countries). If that is right, the reduced version does not fix q14/q15 and the
bridge-detection machinery is not optional. Read the questions first.

---

## 6. What I would not build

- **The unified ingest pass** producing claims, entities and stances from one
  model call. It is elegant and it couples three changes into one prompt,
  against this document's own closing advice. One overloaded prompt degrades
  every job in it. Ingest claims and entities together (both are low-judgment
  extraction of the same shape); leave stance to arc's readers.
- **Bridging-fact generation** at ingest — one call per bridge entity is
  unbounded cost for an unmeasured gain.
- **API-level structured output.** New code near a deadline, replacing a
  regex that currently works.

---

## 7. Open questions, in the order they block work

1. **Do the OECD cross-section questions name their entities?** They live on
   JupyterHub, not in the repo, so I could not read them. This decides whether
   Step 3 is an afternoon or a day.
2. **Does `ministral-3b-2512` hold the `RECORD |` format?** If it does, the
   ingest pass and possibly the whole reader fleet get much cheaper. Testable
   on one chunk for near-zero spend.
3. **Full-document g6 runs on HDR and parks have never been done.** Every
   number above q16–q18 is OECD-only. Before shipping anything, confirm g6 has
   not quietly regressed the two documents where we have history.
