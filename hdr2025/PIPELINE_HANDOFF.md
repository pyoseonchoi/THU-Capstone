# Long-Context Challenge — Pipeline Handoff

Everything needed to resume this work cold: what was built, why, what each
change bought, and what is still broken.

**Two documents are now in play.**

| | Document | Best | Status |
|---|---|---|---|
| **Parks** | `nationalparks_europe.txt` — 93k words, 1.2× window | **v12 at 82.8%** | frozen at user's request |
| **HDR** | `hdr2025.txt` — 211k words, 3.2× window | first run pending | active, in `hdr2025/` |

Started at 38.6%. The parks work is stopped; the HDR pipeline is the live one.

---

## 1. The task

Answer questions about a document larger than the model's context window.
Seven categories, three questions each. Six of the seven are chosen because
top-k retrieval cannot answer them; Needle is the retrieval-friendly control.

### Broker

| | |
|---|---|
| Base URL | `http://80.151.131.52:9839/v1` (OpenAI-compatible) |
| Key | `BROKER_API_KEY` env var, one per group |
| Models | **only** `mistral-small-3-2` (24B) and `ministral-3b-2512` (3B) — anything else 404s |
| Spend | `GET /v1/usage` with the same bearer token |
| Failed calls | cost nothing |

Reachable over the public internet (a dummy key returns `401`, not a timeout),
so Colab works as a backup — it is not VPN-gated.

### Scoring

```
question score = max(0, covered / total_key_points − 0.25 × forbidden_hits)
      quality  = mean over the WHOLE question set
```

Four consequences that drive every prompt in this repo:

- **Always answer.** An omitted question scores zero, same as a wrong one.
- **Never hedge.** Listing alternatives does not state a key point and can trip
  a forbidden claim at the same time.
- **Padding is FREE.** "Extra material neither adds nor subtracts, unless it
  happens to state something on the forbidden list." Being thorough costs
  nothing; being brief loses key points.
- **Verdicts must be singular.** Forbidden claims live in verdicts ("which one
  is absent", "which is largest"), not in supporting detail.

### Submission

One JSON file. `answers[].id` must match exactly; a question may appear at most
once; unknown fields are ignored (so `timing`, `decision` and `evidence` ride
along free). Write the file **inside** the answer loop.

Question sets are distributed as **YAML** — `load_questions()` is a YAML reader.
A set carrying `key_points`/`forbidden` loads fine; only `id` and `question` are
used and extra fields are ignored. The HDR practice set arrived as JSON and had
to be converted; that was the exception, not the rule.

---

## 2. The two ideas that produced every gain

**A. Anything countable is computed in Python. The LLM only does what regex
cannot.**

| Job | Owner |
|---|---|
| Find the document's repeated structure | regex |
| Untangle a scrambled stats box | LLM, then overruled by deterministic pairing |
| Normalize units, count, rank | Python |
| Detect contradictions | Python |
| Decide what is absent | Python |
| Read prose and synthesize | LLM |

**B. Read the document verbatim, in slices sized to the window.**

This is the single highest-value idea in the project and it generalizes by
ratio, not by page count. **Measure `chars / 4` before writing any code.**

| Ratio to window | Strategy |
|---|---|
| ≤ 1× | Put the whole document in one prompt. Nothing else needed. |
| ≤ ~20× | Read every slice for every question. Parks (1.2×) and HDR (3.2×) are both here. |
| ≫ 20× | Compile once, prefilter deterministically, map only candidates. |

Parks: 5 slices → v10→v11 went 62.3% → 80.2%, because our own compression was
destroying the sentence-level facts the judge grades on. Then v13 found that
**slice size controls whether a reader enumerates or generalizes**: 15 slices of
~6k words each made readers list named instances; 5 slices of ~19k made them
write summaries worth zero key points. HDR uses 34 slices at the same ~6k words.

**Corollary that bit three separate times:** every time the store grows,
previously working signals get buried. Always check answer-context size after
adding data.

### On "no RAG"

The rules never banned retrieval — they say the strongest systems *keep* it for
needles and add global machinery alongside. Regardless, this pipeline is
question-independent, and that was **verified empirically**: hashing the
document content sent to every reader across four structurally different
questions gives identical content sets. No
`embed / similarity / cosine / top_k / faiss / vector` anywhere.

Two question-conditioned steps exist, and neither withholds document content:

1. The `NO EVIDENCE` filter discards **model output** after every reader has
   already read its full slice.
2. **HDR only:** `check_question_terms` counts phrases *from the question*
   against the body. It selects, ranks and hides nothing — all 34 readers still
   read all 34 slices — but it is the one thing not byte-identical across
   questions. Off switch: `HDR_TERM_CHECK=off`. Disabling it likely costs the
   three absence questions, since a vocabulary index cannot prove a *trigram*
   absent.

---

## 3. Score history — parks

| Ver | Overall | Absence | Aggr | Contra | Cross | Synth | Needle | Super |
|---|---|---|---|---|---|---|---|---|
| v2 | 38.6 | 27.8 | 0.0 | 16.7 | 33.3 | 53.3 | 83.3 | 55.6 |
| v3 | 47.0 | 63.9 | 46.7 | 38.9 | 0.0 | 60.0 | 63.9 | 55.6 |
| v4 | 55.8 | 52.8 | 63.3 | 2.8 | 66.7 | 46.7 | 100.0 | 58.3 |
| v5 | 61.0 | 44.4 | 75.0 | 47.2 | 33.3 | 46.7 | 100.0 | 80.6 |
| v6 | 67.0 / 60.4 | 16.7 | 75.0 | 63.9 | 53.3 | 60.0 | 100.0 | 100.0 |
| v6 @ temp 0 | 62.5 | 16.7 | 75.0 | 63.9 | 40.0 | 53.3 | 100.0 | 88.9 |
| v7 | 63.3 | 66.7 | 66.7 | 80.6 | 46.7 | 46.7 | 66.7 | 69.4 |
| v8 | 65.4 | 52.8 | 68.3 | 80.6 | 66.7 | 53.3 | 80.6 | 55.6 |
| v9 | 66.0 | 33.3 | 75.0 | 80.7 | 60.0 | 46.7 | 66.7 | 100.0 |
| v10 | 62.3 | 33.3 | 68.3 | 72.2 | 46.7 | 60.0 | 100.0 | 55.6 |
| v11 | 80.2 | 69.4 | 75.0 | 72.2 | 73.3 | 80.0 | 100.0 | 91.7 |
| **v12** | **82.8** | **91.7** | **100.0** | 72.2 | 60.0 | 66.7 | 100.0 | 91.7 |
| v13 | 72.3 | 75.0 | 83.3 | 63.9 | 60.0 | 73.3 | 100.0 | 91.7 |
| v14 | 80.7 | 69.4 | 100.0 | 72.2 | **51.7** | 80.0 | 100.0 | 91.7 |

**Best-of-all-runs ≈ 91.3%.** Every question has been answered well by some
version; the gap is what configurations lose trading wins against each other.

**v12 is the best single configuration and the one to restore if a rollback is
needed.** v13 traded 10 points to fix enumeration; v14 recovered most of it with
the two-block synthesis but never fixed cross-section, which is the standing
weakness at 51.7%.

---

## 4. Version history

| Ver | Change | Result |
|---|---|---|
| **v2** | RAPTOR-lite ordered summary tree. **Makes no structural assumptions** — the last-resort fallback. Produces `tree.json`. | 38.6% |
| **v3** | Anchored on the repeated `"Park in numbers"` box; table and aggregates computed in Python. | Aggregation 0→47 |
| **v4** | Full entry spans capturing post-box prose. Plausibility ranges. | Needle→100 |
| **v5** | Tokenized boxes (NUM/LABEL) + bijection validation. | Contra→47, Super→81 |
| **v6** | Six deterministic indexes; `temperature=0`; timing instrumentation. | 67.0/60.4 |
| **v7** | Word families, term→park locator, absence protocol. | Absence 17→67 |
| **v8** | Two views answered separately, then synthesized. | 65.4% |
| **v9** | A view may declare `NO EVIDENCE`; skip is mechanical. | Superlative→100 |
| **v10** | Every printed stat promoted; bigrams down to 2. | 62.3% |
| **v11** | **Read the raw text in 5 slices.** Deterministic box pairing, headline answers, designation table. | **80.2%** |
| **v12** | Additive synthesis (union, no length cap), `never_mentioned` boolean, contradiction dedupe. | **82.8% — best** |
| **v13** | 15 slices instead of 5; readers return instance LISTS, not prose. | 72.3% — fixed enumeration, broke 4 questions |
| **v14** | **Two-block synthesis**: computed values authoritative and un-overrulable, text evidence combined by union. Anti-invention rule. | 80.7% |
| **h1** | HDR document. Parks compiler replaced wholesale; v14's answering half kept. | pending |

**Why v13 dropped 10 points, and the rule that came out of it:** with 15 readers
instead of 5 there are 15 chances at invention, and v12's additive rule ("every
detail any reader gave MUST appear") faithfully preserved a hallucinated park
name that displaced a correct computed value. **Union is safe when every draft
is a grounded quote and dangerous when one draft is invention.** v14's fix —
rank the sources instead of pooling them — is the single most transferable
prompt-level lesson in this project.

---

## 5. Files

### Parks (root folder — frozen)

```
pipeline_ed2.py   LLM plumbing. get_client (lazy), call_llm(+temperature),
                  call_llm_json (JSON repair + retry), preprocess_document,
                  load_questions. EVERYTHING imports this.
pipeline_v2.py    RAPTOR-lite tree. Document-agnostic. THE LAST-RESORT FALLBACK.
pipeline_v3.py    normalize_park (+ _is_height_label), compute_aggregates.
pipeline_v4.py    find_park_entries, implausible, recheck_park, load_narratives.
pipeline_v5.py    tokenize_box, find_contradictions, SCOPE_GROUPS.
pipeline_v6.py    coverage index, superlative scanner, dated events, country
                  resolver, timing helpers, ANSWER_TEMPERATURE, STOPWORDS.
pipeline_v7.py    word families, term locator, digit-aware coverage index.
pipeline_v8.py    the two views + first synthesis.
pipeline_v9.py    NO_EVIDENCE declaration + mechanical skip.
pipeline_v10.py   park_stats_index, bigram threshold 2, plural lookup rule.
pipeline_v11.py   raw-text chunking, deterministic box pairing, headline answers.
pipeline_v12.py   additive synthesis, never_mentioned flag. BEST PARKS SCORE.
pipeline_v13.py   15 slices, list-format readers.
pipeline_v14.py   two-block synthesis, anti-invention rule.
pipeline.py       STALE duplicate of an early ed2. Do not import.
snapshots/        v6_pre_temperature/ — rollback copy.
submission/       graded submission files.
```

Each version imports the ones below it — **v14 needs ed2 and v3–v13 present.**

### HDR (`hdr2025/` — active)

```
hdr2025.txt              the converted document, 211,447 words, 314 pages
hdr_compile.py           THE NEW COMPILER. Annex table 1, Contents lists,
                         body-only coverage, figure captions. No LLM, <1s.
pipeline_h1.py           v14's answering half + hdr_compile. Self-contained:
                         imports only ed2, v6, v9, hdr_compile.
hdr_questions.yaml       21 questions (converted from the supplied JSON)
VERIFIED_ANSWERS.md      11 of 21 answers derived in Python and cross-checked.
                         The substitute for the missing key_points.
parks_lineage/           v10–v14, kept only so the import chain stays intact.
```

`pipeline_h1.py` deliberately does **not** import v10–v14 — it re-implements
`answer_question` and `run_submission` directly rather than monkeypatching a
five-deep chain, which is why the parks versions could be moved aside.

### Running

```bash
export BROKER_API_KEY=your_group_key      # same shell as the run
nohup python -u pipeline_h1.py > run.log 2>&1 &
tail -f run.log
```

`nohup` because JupyterHub's idle culler can kill the server.

Environment overrides: `HDR_DOCUMENT`, `HDR_QUESTIONS`, `SUBMISSION_FILE`,
`HDR_CHUNKS`, `HDR_WORKERS`, `HDR_TERM_CHECK`, `ANSWER_TEMPERATURE`.

Submission filenames run **two ahead** of the pipeline number (`pipeline_h1` →
`submission_h3.json`). The grader ignores filenames.

---

## 6. PDF → text conversion — hard-won, fully transferable

The HDR arrived as a 324-page PDF. **This section is the most reusable thing
learned today**, because every future document will arrive the same way.

### Never trust one extraction mode

| Mode | Reading order | Tables | Verdict |
|---|---|---|---|
| `pypdf` | ok | ok | **Unusable.** Splits words at kerning boundaries: `New Y ork` 39× vs `New York` 2×. Found 117 "artificial intelligence"; pdftotext found 183 — **silently losing a third of every term count.** |
| `pdftotext -layout` | **broken** — two-column pages emit line-alternating, so sentences interleave | **correct** | tables only |
| `pdftotext` (plain) | **correct** — reflows columns, de-hyphenates | **broken** — tables collapse to a bare column of names with numbers detached | prose only |

**The answer is a per-page hybrid.** Classify pages by numeric-field density,
then take tables from `-layout` and prose from plain mode. For the HDR: pages
288–319 (statistical annex) and 323–324 (HDI rank key) from `-layout`, 11–13
(Contents) from `-layout` **with a column split**, everything else plain.

### Traps found the hard way

- **Verify with term counts, not eyeballs.** The pypdf defect is invisible when
  reading a page and fatal to an exact-count index. Compare candidate
  extractions on the same handful of terms.
- **Do not de-hyphenate blindly.** `-layout` produced 2,482 line-end hyphens,
  but `compla-` + `from` are two different *columns*, not one word. Plain mode
  had **zero** — pdftotext already de-hyphenates when it reflows.
- **Splitting columns on the widest blank gutter is wrong.** On HDR page 13 the
  widest middle gap sits between the right column's *identifier* and its
  *title*; cutting there stripped every identifier off the TABLES list and the
  section parsed as empty. **Split on the identifier column positions instead** —
  they cluster tightly (0 and ~90 on every Contents page).
- **Strip zero-width spaces** (415 in the HDR) and non-breaking hyphens. Both
  break exact matching invisibly.
- **Letter-spaced headings** arrive as `F I G U R ES`, `TABL ES`,
  `S POTL I G H TS`. Compare with whitespace removed. Column-splitting also
  glues the neighbouring column's page number on (`62      BOXES`), so strip
  digits off both ends too.
- **Emit `[page N]` markers.** They give every fact a citable address and stop
  the chunker fusing one page onto the next.
- **Known residual defect:** dense multi-column endnote/reference pages
  (HDR ~217–225, ~279–286, about 4%) interleave in every mode. They are
  citations, so low value — but do not write questions against them.

---

## 7. The HDR compiler — what a new document needs

v14's **answering** half is fully document-agnostic. Its **ingest** half is not:
`find_park_entries` and the `"Park in numbers"` box pairing find nothing in the
HDR, so running v14 unchanged would silently lose Aggregation, Superlative and
Needle — the three categories it is best at. This is the single most important
thing to check when the document changes.

`hdr_compile.py` replaces it with four parsers, all verified:

| Component | Answers | Verification |
|---|---|---|
| Annex Table 1 → 193 ranked rows | h01, h04–h06 | bands sum to 193; HDI ranges sit exactly inside the report's published cutoffs with zero leakage; 193 distinct countries, no duplicates |
| Contents lists | h02, h03 | Contents identifier count == body caption count, **identical ID sets** (67 figures, 19 boxes, 6 tables) |
| **Body-only coverage index** | h07–h09 | see below |
| Figure caption index | h13–h15 | resolves the exact figures the questions cite |

### The body/references split is the new idea

The HDR carries an 87-page reference section. h07 asks which of four subjects is
"never mentioned". `metaverse` occurs **exactly once in the whole file** — inside
the title of a cited ITU report. A whole-document index says "mentioned", which
is the wrong answer. Counting pages 1–233 only says "never", which is right.

**Generalizes to any document with a bibliography:** a term appearing only in a
cited work's title is not a subject the document discusses.

### Verified HDR answers — do not re-derive

```
h01  193 ranked entries: 74 very high / 50 high / 43 medium / 26 low
h02  BOXES 19, SPOTLIGHTS 12, TABLES 6
h03  67 figures; Overview 8, Ch1 9, Ch2 4, Ch3 19 (most), Ch4 6, Ch5 9, Ch6 12
h04  Iceland 0.972 (rank 1)
h05  San Marino 85.7 years
h06  Liechtenstein 166,812 (2021 PPP $)
h07  the metaverse   (body 0, references 1)
h08  digital twins   (0, 0)
h09  universal basic income (0, 0)
h19  PDF ISBN 9789211542639  (print ISBN 9789211576092 — different)
h21  12 orders of magnitude  (body sentence AND Figure 1.6 caption agree)
```

Full working in `hdr2025/VERIFIED_ANSWERS.md`.

---

## 8. Traps in the parks document — verified, do not re-derive

**The stats boxes are scrambled by PDF conversion.**

| Box | Trap |
|---|---|
| 24 Hortobágy | area is **820**; 135,000 is cranes. "**Peak** annual population" read as a summit → a 135,000 m peak |
| 10 Cairngorms | area 4528, summit **1309** (Ben Macdui). A *swap* passes any plausibility range |
| 27 Killarney | one label token holds TWO labels; the head absorbed the tail's 4500 → a 4,500 m Irish summit |
| 3 Aigüestortes | run-on label hid a real 3,033 m summit |
| 26 Jotunheimen | `1151 Area covered (sq miles)` — the **only** park not in sq km |
| 58 Vatnajökull | 13,600 km² — the largest |
| 43 Pirin | 1380-year-old tree — a box stat, not prose (v13 hallucinated "Rila" here) |
| 14 Curonian Spit | 98 km, 52 of which in Lithuania |

**Correct box pairing needs all three of:** global nearest-first assignment (not
label order), label cleaning at the unit bracket, and splitting merged label
tokens.

**Other verified facts:**

- Exactly **8** parks report a summit ≥3000 m.
- **Paklenica's entry never contains the word "Croatia"** — why d03 counts 2.
- `poaching 0 | natura 0 | glaciation 0 | wartime 0 | glacier 63 | glaci* 173`
- `glaciation` 0 but `glacier` 63 → word families are mandatory.
- `accessible` 39 (travel sense) but `wheelchair` 0 → head words mislead.

---

## 9. Run-to-run variance

Runs #19 and #21 were the **same code over the same data** and scored **67.0%
and 60.4%**. 15 questions scored identically — every one backed by a computed
value. The 6 that moved were all free prose.

`ANSWER_TEMPERATURE = 0.0` fixes this. **Single-run numbers before v6 carry ±7
points of noise.** Category moves under ~20 points between adjacent versions may
be noise; with 3 questions per category, one question swings a category by 33.

**Rolling back does not restore a score** — 67% was the luckier of two samples.
The only artifact that preserves a score is the graded file itself.

---

## 10. Method that works — use this before changing anything

**Diagnose against the raw text locally, with no LLM calls, before touching the
pipeline.** Every real fix in this project came from a 20-line script:

- Regexing all 60 boxes revealed the 135,000-cranes mispairing.
- Tokenizing box 10 showed a swap no plausibility range could catch.
- Counting terms proved `poaching` and `natura` genuinely absent.
- Comparing extraction modes on term counts exposed the pypdf defect.
- Measuring the file proved the parks document is only 1.2× the window — the
  single most valuable measurement of the project.

**Dry-run `__main__` with the LLM mocked before spending a call.** v7 shipped a
`KeyError` in its entry point because dry runs mocked around `__main__`. The HDR
pipeline was validated end-to-end this way — 756 calls, largest prompt 14.6k
tokens, all failure paths exercised — for zero broker spend.

**Read the judge's per-key-point feedback.** It converts "Absence is weak" into
"d09 wants four things and we answer one" — a different problem with a different
fix. Ask for it every run.

**Rules of thumb learned the hard way:**

- Change **one thing** per scored run. v4 changed three and hid a regression.
- After adding data to the store, check the answer-context size.
- Never let a fallback be silent — `decision`, `country_uncertain` and the
  truncation notes all exist so a degraded answer is visible.
- When two components disagree, do not pick by fiat. v9's "absence belongs to
  the index reader" rule discarded a correct answer. Resolve against data.
- Set the client to `timeout=300, max_retries=0`. The SDK default is 600s × 2
  retries, which hung one question for 15 minutes.

---

## 11. Competition day

The private document is a **different file with different questions**, delivered
as YAML.

**Playbook:**

1. **Convert the PDF properly** — §6. Compare two extraction modes on term
   counts before trusting either.
2. **Measure the document.** `chars / 4`. Pick the regime from §2B and set
   `HDR_CHUNKS` so each slice is **~6k words**, not by slice count.
3. **Look for a repeated per-entry structure** and for a bibliography boundary.
   Budget time for this before running anything.
4. **If found:** write a compiler for it, following `hdr_compile.py`. Verify
   every parser two independent ways before trusting a number.
5. **If not found:** the reader half works alone with no compiler at all. That
   is the floor, and it is a decent one.
6. **Transfers unchanged, document-agnostic:** raw chunking, the two-block
   synthesis, the anti-invention rule, `NO EVIDENCE` + mechanical skip,
   `call_llm_json`'s repair/retry layer, `temperature=0`, the coverage index,
   word families, dated events.
7. Write incrementally, upload early, re-upload as often as you like.

Cost has never been a constraint: ~$6 spent against another team's $23.71.

---

## 12. Open issues

1. **Cross-section, 51.7% in v14** — the standing weakness. d13/d14/d15 lose
   dates, qualifiers and cross-border specifics. Not diagnosed.
2. **No local scoring on the HDR set** — it ships no `key_points`/`forbidden`.
   `VERIFIED_ANSWERS.md` covers 11 of 21; the other 10 are unmeasurable until
   the grader replies. If the final YAML carries key points, wire the scorer
   back in immediately.
3. **HDR cost/time unmeasured** — 756 calls vs the parks run's 336, so expect
   ~2.2×, roughly 1–1.5 hours. Confirm on the first real run.
4. **d09 still names "wartime damage"** instead of poaching; **d05 omits "in
   France"** (parks, frozen).

---

## 13. 한국어 요약 — 지금까지의 과정

### 출발점

JSON 파싱 에러 하나로 시작했습니다. 작은 모델(`ministral-3b`)이 괄호가 안 맞는
JSON을 뱉어서 파이프라인 전체가 죽었죠. 여기서 배운 게 이후 전부의 토대가
됐습니다 — **작은 모델은 반드시 실패한다, 실패를 전제로 설계해야 한다.**

### 핵심 통찰 여섯 가지

**1. 셀 수 있는 것은 파이썬이 센다.** (v3, 0% → 47%)
"스페인에 공원이 몇 개인가"는 요약문을 읽고 세는 문제가 아니라 계산입니다.
LLM에게 60개 행을 세게 하면 틀립니다. 파이썬 `max()`는 안 틀립니다.

**2. 컨텍스트 창은 제로섬이다.** (세 번 데임)
absence를 위해 색인을 넣으면 needle이 죽고, needle을 살리면 absence가 죽었습니다.
"저장소에 뭘 추가할 때마다 잘 되던 신호가 묻힌다."

**3. temperature를 안 정하면 점수는 노이즈다.** (±7점)
똑같은 코드를 두 번 돌려 67.0%와 60.4%가 나왔습니다. 계산값 기반 문항 15개는
완전히 동일했고, 자유 서술 문항 6개만 흔들렸습니다.

**4. 문서가 컨텍스트보다 조금 클 뿐이었다.** ← 전환점 (62.3% → 80.2%)
우리는 문서를 90k 토큰짜리 저장소로 **압축**하고 있었는데, 5조각으로 자르면
원문을 통째로 읽힐 수 있었습니다. 채점 기준이 요구하던 문장 단위 사실들은
우리가 **입수 단계에서 버린** 것이었습니다.

**5. 조각 크기가 "나열이냐 요약이냐"를 결정한다.** (v13)
19k 토큰을 쥔 리더는 요약합니다("전통적 토지 이용이 보호와 공존한다" — 사실이지만
0점). 6k 단어를 쥔 리더는 나열합니다("아비스코의 사미족 순록 방목"). 그래서
조각 개수가 아니라 **조각당 단어 수(~6천)**를 기준으로 자릅니다.

**6. 합집합은 위험할 수 있다.** (v13 → v14, 이게 가장 값진 교훈)
v12의 "모든 리더의 디테일을 다 살려라"는 규칙은 리더가 5명일 때 옳았습니다.
15명이 되자 15번의 환각 기회가 됐고, 한 리더가 지어낸 "Rila 산맥"이 계산된
정답을 밀어냈습니다. **합집합은 모든 초안이 인용일 때만 안전하고, 하나라도
창작이면 위험합니다.** v14의 해법은 합치지 말고 **서열을 매기는 것** — 계산된
값은 절대 뒤집을 수 없는 블록, 텍스트 증거는 그 아래 합집합 블록.

### HDR 2025 (새 문서)

324쪽 PDF를 텍스트로 변환하는 것부터 시작했습니다. 여기서 배운 것:

- **pypdf는 쓸 수 없습니다.** 단어를 쪼갭니다(`New Y ork`). "artificial
  intelligence"를 183번 중 117번만 찾았습니다 — 3분의 1을 조용히 잃습니다.
- **`pdftotext -layout`은 표에는 맞지만 2단 본문에서 줄이 교차**합니다.
- **`pdftotext` 기본 모드는 본문에는 맞지만 표가 무너집니다.**
- 답은 **쪽 단위 혼합**입니다. 표는 `-layout`, 본문은 기본 모드.

**본문/참고문헌 분리**가 새로운 아이디어입니다. HDR은 참고문헌만 87쪽입니다.
h07은 "한 번도 언급되지 않은 주제"를 묻는데, `metaverse`는 문서 전체에서 딱
한 번 — 인용된 ITU 보고서 **제목 안에서** 나옵니다. 전체를 세면 "언급됨"(오답),
1~233쪽만 세면 "언급 안 됨"(정답).

21문항 중 **11문항을 LLM 없이 파이썬으로 확정**했고, 각각 두 가지 독립적인
방법으로 교차 검증했습니다.

### 결과

**38.6% → 82.8%** (v12, 파크스 최고점). 가장 중요한 건 **최저 카테고리가
27.8%에서 60%대로** 올라간 것입니다.

| | v2 | v12 | v14 |
|---|---|---|---|
| Needle | 83.3 | 100.0 | 100.0 |
| Aggregation | 0.0 | 100.0 | 100.0 |
| Superlative | 55.6 | 91.7 | 91.7 |
| Absence | 27.8 | 91.7 | 69.4 |
| Contradiction | 16.7 | 72.2 | 72.2 |
| Global synthesis | 53.3 | 66.7 | 80.0 |
| Cross-section | 33.3 | 60.0 | 51.7 |

### 가장 값진 방법론

**바꾸기 전에 원문을 직접 뒤져라.** 실제로 고쳐진 것들은 전부 LLM 호출 없이
20줄짜리 스크립트로 찾았습니다 — 학두루미 135,000마리를 면적으로 읽은 것,
케언곰스의 4528(면적)과 1309(높이)가 뒤바뀐 것, `poaching`이 정말 0회인 것,
pypdf가 단어를 쪼개고 있던 것, 그리고 문서 크기 측정.

**브로커 호출 전에 `__main__`을 모의 실행해라.** HDR 파이프라인은 756번의
호출 경로와 모든 실패 경로를 **비용 0원**으로 검증한 뒤에 처음 돌렸습니다.
