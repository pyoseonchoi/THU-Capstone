# g13 — contradiction: what the diagnosis missed, and what fixes it

All numbers below are deterministic replays of the g12 run's own reader lines.
No broker calls were spent producing any of them.

## The diagnosis needed correcting

The brief said: candidates are handed over as a loose pile and the model pairs
them freely; detection is not the problem. Rebuilding the pool from the run's
own evidence blocks says otherwise.

| | claimed | actual |
|---|---|---|
| **d12** | pile → free pairing | Pool held **2** candidates and #2 *was* the answer. The **adjudicator model rejected both**. The block went out empty and synthesis invented Swiss 170.3 vs 169 from reader lines. |
| **d11** | pile → free pairing | Pool held 15 candidates and **the answer was not among them**. The adjudicator kept exactly **one** — no free pairing occurred. |
| **d10** | needs a rank detector | Right, but for a reason step 4 alone could not reach. |

**The missing half was always the claim, never the figure.** Both records
needed for d11 were present and verified (Hardangervidda 3430,
Nordvest-Spitsbergen 3583), as were d10's (Snowdon 1085, Ben Macdui 1309). What
was absent was the sentence to set against them — and both are in the document:

- *"the high plateau of Hardangervidda that dominates **Norway's largest national park**…"*
- *"At 1085m, **Britain's second-highest peak** has been a testing ground since 1798…"*

Readers emitted neither. For Hardangervidda they wrote *"northern Europe's most
accessible wilderness areas"*; for Snowdon, *"the highest mountain in Wales"* —
six times. The second sentence never says the word *Snowdon*, which is likely
why.

Two smaller corrections. The d11 peak-vs-area pairing and the d10 "978 < 1085"
sentence were **not** mis-paired candidates: the computed block held something
else entirely in both cases, and the model wrote them from reader lines. That
makes step 5 more important than scoped, not less — forbidding invention in the
prompt demonstrably did not hold.

## What was built

`find_text_claims()` reads superlative and rank claims out of the raw document,
exactly as `find_label_unit_outliers()` already reads units out of it — "immune
to what any reader chose to report". Zero model calls.

Seven flags, all pure Python, `G13_FIXES=` restores g12 exactly:

| flag | what it does |
|---|---|
| `textclaims` | claims from the document, not from reader recall |
| `namescope` | an entity with no country record is scoped by its own name |
| `rank` | a rank claim checked against the ordering the document states |
| `strictscope` | a claim naming a country may only be refuted from inside it |
| `attrmatch` | both halves of a pair must measure the same thing |
| `route` | candidates matched to the question by keyword overlap |
| `onepair` | one validated pair, comparison sentence pre-written in Python |

## Result — same records, same claims, new assembly

| | g12 | g13 |
|---|---|---|
| d10 | no candidate survived validation | **Snowdon 1085 · Cairngorms group claim · Ben Macdui 1309** — 4/4 key terms |
| d11 | no candidate survived validation | **Hardangervidda 3430 vs Nordvest-Spitsbergen 3583** — 4/4 |
| d12 | no candidate survived validation | **Jotunheimen 1151 sq miles vs 26 entries in sq km** — 3/3 |

The sentences the model now receives, already arithmetic-checked:

> Snowdon is described as Britain's second-highest peak at 1085, but the
> document also states that five of Britain's six highest summits are in
> Cairngorms (page 46) — so at least 5 rank above it and it cannot be number 2.
> Ben Macdui is given as 1309m (page 46).

> Hardangervidda National Park is described as Norway's largest national park at
> 3430 (sq km) (page 95), but NordvestSpitsbergen National Park is given 3583 sq
> km (page 147), which is larger.

> Jotunheimen National Park reports its Area covered as 1151 sq miles, while 26
> other entries give the same measurement in sq km.

## Leave-one-out ablation

| flag removed | d10 | d11 | d12 |
|---|---|---|---|
| *(none — full set)* | 4/4 | 4/4 | 3/3 |
| `textclaims` | 0/4 | 0/4 | 3/3 |
| `rank` | 0/4 | 4/4 | 3/3 |
| `strictscope` | 4/4 | 0/4 | 3/3 |
| `namescope` | 4/4 | 0/4 | 3/3 |
| `route` | 4/4 | 0/4 | 0/3 |
| `attrmatch` | 4/4 | 4/4 | 3/3 |

Every flag is load-bearing except `attrmatch`, which is honest to report: it is
the brief's step 2, and it cannot fire on these three because the peak-vs-area
pairing it guards against came from reader lines rather than from a candidate.
It is kept as the correct invariant, not as a measured gain.

`onepair` alone fixes d12 — the fallback plan in the brief was right about that
one question, and for the stated reason.

## Shared code paths touched

Asked explicitly, and answered by `ledger_isolation.py`, which byte-compares
every prompt g12 and g13 build for all 21 questions on three documents:

| document | contradiction | other 18 questions | call delta |
|---|---|---|---|
| PARKS | changed (intended) | **all identical** | **−3** |
| OECD | changed (intended) | **all identical** | +0 |
| GEM | changed (intended) | **all identical** | +0 |

Because the synthesis prompt embeds the computed block, identical prompts prove
the Python arithmetic upstream is identical too — not just the wording.

Modified functions: `entity_scopes` (guarded by `namescope`), `check_claims`
(guarded by `strictscope`), and the `mode == "ledger"` branch. Everything else
is new code reachable only from that branch. Absence, needle, global synthesis,
cross-section, aggregation and superlative are untouched.

The −3 on parks is the adjudicator call removed: a pair validated in Python is
no longer sent to a model to be judged, which is what lost d12.

## Caveat

These are replays. They prove the assembly stage now selects and states the
right pair **given the records the last run produced**. A live run re-extracts,
and extraction varies between runs. The claims come from the document and are
stable; the records are not.
