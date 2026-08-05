# Pipeline Handoff — Long-Context QA Without RAG

**Read §0 and §1 before writing any code.** This document exists so the work can
be resumed cold, by a different person or a different assistant, without
repeating mistakes that have already cost us two rewrites.

Last updated: 2026-08-05, after the first real broker run of `pipeline_g1.py`.

---

## 0. The goal — state it back before you start

Build **one pipeline that reads any `.txt` document of any size** and answers a
supplied question file, **without RAG**.

On competition day the organizers hand us two files: the document as `.txt` and
the questions. There is **no PDF conversion** on the critical path. The document
is **unknown until ~60 minutes before the deadline**, and each team gets **one
evaluation run**.

The deliverable is **not** a model for a specific document. If a constant in the
code names one document — a page number, a section title, a box label — that is
a bug, not an optimization.

### The standing failure mode: fitted compilers

This has now happened twice, and both times it scored well and then generalized
to nothing.

| What | How it was fitted | What it finds on the other document |
|---|---|---|
| `pipeline_v3`–`v14` | anchored on the literal `"Park in numbers"` box | nothing |
| `hdr_compile.py` | Annex Table 1 pinned to pages 288–292, Contents to 11–13, HDI band names hardcoded | nothing |

**The dangerous one, verified by running it:** absence questions were keyed to
`[page N]` markers that *our own PDF converter* wrote. A `.txt` supplied by the
organizers has none. With no markers, the body resolves to an empty string,
every candidate term counts zero, and the pipeline confidently answers "the
document never mentions X" about a term occurring 54 times. That is a
*forbidden hit*, not a missed point — the worst outcome the scoring formula
allows.

**Rule:** structure may be **discovered at runtime**, never **assumed**.

---

## 1. Hard constraints

### Broker

| | |
|---|---|
| Base URL | `http://80.151.131.52:9839/v1` (OpenAI-compatible) |
| Key | `BROKER_API_KEY` env var — **one key per group, shared by all 4 members** |
| Models | **only** `mistral-small-3-2` (24B) and `ministral-3b-2512` (3B) — anything else 404s |
| Spend | `GET /v1/usage` with the same bearer token |
| Failed calls | cost nothing |
| Client settings | `timeout=300, max_retries=0` — the SDK default of 600s × 2 hung one question for 15 minutes |

Reachable over the public internet (a dummy key returns `401`, not a timeout),
so Colab works as a backup — it is not VPN-gated.

**Because the key is shared, two members running at once confound both the
quality measurement and the Performance criterion. Serialize runs.**

### Scoring

```
question score = max(0, covered / total_key_points − 0.25 × forbidden_hits)
      quality  = mean over the WHOLE question set
```

Four consequences that drive every prompt in this repo:

- **Always answer.** An omitted question scores zero, same as a wrong one.
- **Never hedge.** Listing alternatives states no key point and can trip a
  forbidden claim at the same time.
- **Padding is FREE.** Extra material neither adds nor subtracts unless it
  states something forbidden. Being thorough costs nothing; being brief loses
  key points.
- **Verdicts must be singular.** Forbidden claims live in verdicts ("which one
  is absent", "which is largest"), never in supporting detail.

**Corollary that the latest run proved expensive:** an invented number is worse
than a missing one. It can trip a forbidden claim, and human graders read the
answers.

### Grading criteria (5)

Quality · Performance (tokens/time) · Innovation · Presentation · Project
Management. Quality is not the only axis — a pipeline that answers well but
cannot explain its architecture loses on two of the other four.

### Submission

One JSON file. `answers[].id` must match the question ids exactly; a question may
appear at most once; unknown fields are ignored, so `timing`, `decision` and
`evidence` ride along free. **Write the file inside the answer loop**, so a
crash leaves a partial submission rather than nothing.

Question sets are distributed as **YAML**. `load_questions()` is a YAML reader,
and JSON parses through it unchanged — both `hdr_questions.yaml` and
`hdr2025_fullscan_questions.json` load to identical structures. Only `id`,
`question` and `category` are used; extra fields like `key_points`/`forbidden`
are ignored.

---

## 2. The current deliverable: `pipeline_g1.py`

One pipeline, any `.txt`, no per-document fitting. This is the live line of work.
Everything before it (`v3`–`v14`, `h1`, `h2`, `hdr_compile.py`) is history.

### Files needed — this is the whole list

```
pipeline_g1.py     the pipeline
pipeline_ed2.py    its only import: get_client, preprocess_document, load_questions
<document>.txt     supplied on the day
<questions>.yaml   supplied on the day (.json also works)
```

Python packages: `openai`, `pyyaml`. Nothing else. **Do not copy the rest of the
folder** — `hdr_compile.py` and `pipeline_v3`–`v9` are the fitted versions and
carrying them forward is how the failure mode returns.

### Architecture

```
load_document
  └─ measure                      chars/4 → tokens; ratio to window
  └─ discover_structure           page markers IF present; bibliography boundary
  └─ auto_chunk_count             words / WORDS_PER_READER   ← derived, never fixed
  └─ chunk_text                   verbatim slices, ~6k words, 300-word overlap

per question → question_mode(category) routes to ONE of three paths:

  ABSENCE   settle_absence()                          0 model calls
            candidate_list → question_subjects → subject_coverage
            → _occurrences (word-boundary) / _family_present (stem)

  TALLY     every reader emits  RECORD | entity | attribute | value | where
            → parse_records → tally() in Python
              (dedupe, unit conversion, threshold, max/min, mixed-unit flag)
            → 1 synthesis call

  PROSE     every reader extracts verbatim evidence
            → 1 synthesis call

wrapped around every model call:
  Throttle (global adaptive rpm: wait / penalize / relax) + jitter
  NO EVIDENCE declaration + mechanical skip
  resume from partial submission
  DEADLINE_MINUTES cutoff
```

**Every reader reads every slice.** That is the "no RAG" property, and on the
older pipeline it was verified empirically: hashing the document content sent to
readers across four structurally different questions gave identical content
sets. No `embed / similarity / cosine / top_k / faiss / vector` anywhere.

### The document-agnostic mechanisms (what replaces the hardcoding)

| Was hardcoded | Now discovered |
|---|---|
| chunk count | `words / 6000` |
| bibliography at page 233 | heading past 45% of the file **plus** a citation-density test (≥15 year-citations or links in the next 5,000 chars) |
| `[page N]` markers required | optional; absence falls back to whole-document counting |
| head-word term matching | word families via 5-char stem — `glaciation` scores 0 but `glacier` occurs 118× |
| substring matching | word-boundary regex — `natura` was matching inside `natural` |
| unit assumptions | parse 3/2/1-token units, normalize before comparing |
| prompts naming the document | prompts name nothing |

### Settings

All are environment variables; the defaults are in the CONFIG block at the top
of the file.

| Var | Default | Note |
|---|---|---|
| `DOC` | `hdr2025.txt` | the `.txt` |
| `QUESTIONS` | `hdr_questions.yaml` | `.json` also works |
| `SUBMISSION_FILE` | `submission_g1.json` | written incrementally |
| `TEAM` | `group_c` | |
| `MODEL` | `mistral-small-3-2` | |
| `TEMPERATURE` | `0` | **do not raise** — see §6 on variance |
| `WORDS_PER_READER` | `6000` | the enumerate-vs-summarize threshold |
| `CHUNK_OVERLAP` | `300` | |
| `WORKERS` | `3` | **use 2** — the group key is shared |
| `RPM` | `90` | throttle start; it self-adjusts |
| `ATTEMPTS` | `8` | |
| `DEADLINE_MINUTES` | unset | set it on competition day |

### Running

```bash
export BROKER_API_KEY=your_group_key          # same shell as the run
DOC=hdr2025.txt QUESTIONS=hdr_questions.yaml SUBMISSION_FILE=submission_g1.json \
WORKERS=2 nohup python -u pipeline_g1.py > run_g1.log 2>&1 &
tail -f run_g1.log
```

`nohup` because JupyterHub's idle culler kills the server.

**Resume is automatic.** A re-run reads the existing submission and skips
completed answers, so an interrupted or partly-failed run costs only the
questions that did not finish.

Cost has never been a constraint: ~$6 spent against another team's $23.71.

---

## 3. Latest measured result — g1 on HDR 2025

`submission_g1.json`, scored against `hdr2025_answer_key.yaml` (frozen before
the run, reused unchanged).

**quality = 0.5635, sum 11.83 / 21**

| category | mean |
|---|---|
| absence | **1.00** |
| cross_section | 0.833 |
| global_synthesis | 0.806 |
| needle | 0.667 |
| contradiction | 0.639 |
| aggregation | **0.00** |
| superlative | **0.00** |

**Shape of the result: 14 questions average 0.845; 7 questions score exactly
0.00. Every single zero is "extract a specific value out of a table."** That is
one bug area, not seven.

### What works

- **Absence 3/3 at 1.00 with zero model calls**, on a document it was never
  tuned for. The guarded Python fast path is the strongest component we have.
- **The `RECORD` format works.** The main open risk was that Mistral would
  ignore the schema and `tally()` would receive nothing. It received plenty —
  the content was wrong, not the format. That is a far more tractable problem.

### What fails — three distinct bugs, verified against the source

1. **Wrong table** (h01, h02, h03, h05). `RECORD_PROMPT` never constrains
   readers to the table the question names, and this document has seven similar
   composite-index tables. `tally()` cannot tell that records came from the
   wrong one. **The questions name their own scope** ("In Statistical Annex
   Table 1…") and `tally()` ignores that entirely.
2. **Wrong column** (h04: Norway / 0.995). `0.995` occurs 8× in the source, but
   as a repeated column value in a *different* table. The number is real; the
   binding is wrong. This is the same whitespace-aligned-table scramble that
   produced the parks box mispairings.
3. **Invented digits** (h06: Qatar / 148,063). The string `148,063` occurs
   **zero times** in `hdr2025.txt`. Qatar's actual row reads `0.886 82.4 13.1`.
   There is no provenance check anywhere in the pipeline.

**h19 is bug 1 in miniature:** two readers disagreed about which ISBN was the
PDF one, and the dedupe rule — *keep whichever record has the longer `where`
string* — picked arbitrarily. When two readers disagree about the same entity
and attribute, that is a conflict signal, not something to resolve by string
length.

### Do not trust the score as an upper bound

h14 and h18 scored 1.00 and 0.75 while containing numbers that could not be
found anywhere in the source ("200/180/160 models", "3.0" skills penetration,
several precise percentages). They tripped no forbidden claim only because the
frozen key did not anticipate those exact fabrications. **0.5635 is an
overestimate.** The key was frozen before the run and must not be amended
retroactively — report this as a narrative caveat, not a score adjustment.

### Comparison to the fitted pipeline

The earlier fitted run (`submission_h4.json`, pipeline `h2`) scored ~0.95 on
this document, because `hdr_compile.py` knew Table 1 sits on pages 288–292 of
*this* file. That knowledge is worth exactly zero on the private document. The
lesson from g1 is that **the generic tally is under-specified, not that generic
tally is impossible.**

---

## 4. Next changes — in priority order

1. **Provenance check (biggest win, fully generic).** Require every `RECORD` to
   carry the verbatim source line, then verify in Python that the number
   actually appears near the entity name in the raw text before the record
   enters the tally. Pure string matching, no model, no document knowledge.
   Would have killed Qatar/148,063 *and* Norway/0.995 *and* caught the h14/h18
   fabrications the answer key missed.
2. **Question-scope filtering.** Extract locator phrases from the question text
   ("Statistical Annex Table 1", "Figure O.1") and filter records whose `where`
   does not match. Uses the *question*, not the document — so it stays generic.
3. **Conflict detection instead of length-based dedupe.** Same entity + same
   attribute + different value = surface the conflict or take a majority vote
   across readers. Never resolve by `len(where)`.
4. Reader-prompt guard against chart furniture (axis ticks `0 20 40 60 80` are
   not quotable facts). Partially done in `PROSE_PROMPT`.

### Known open items outside the code

- `hdr2025_answer_key.yaml` h10's 4th key point ("4.5 times lower") has **zero
  source support** — it is unverifiable and should be removed.
- h09's key point says "social dialogue **on** algorithmic management"; the
  source says "**over**".
- The answer key's `forbidden` lists were written after `submission_h4` and
  encode that submission's specific errors, which biases it against its own
  author. Have the other team member audit it before any head-to-head.

---

## 5. Verified facts — do not re-derive

### HDR 2025 (`hdr2025.txt`, 211,447 words)

```
h01  193 ranked entries: 74 very high / 50 high / 43 medium / 26 low
h02  BOXES 19, SPOTLIGHTS 12, TABLES 6
h03  67 figures; Overview 8, Ch1 9, Ch2 4, Ch3 19 (most), Ch4 6, Ch5 9, Ch6 12
h04  Iceland 0.972 (rank 1)          — NOT Norway; Norway's real HDI is 0.970
h05  San Marino 85.7 years           — Hong Kong 85.5 is the naive-retrieval trap
h06  Liechtenstein 166,812 (2021 PPP $)  — Qatar's real GNI is 105,353
h07  the metaverse            (body 0, references 1)
h08  digital twins            (0, 0)
h09  universal basic income   (0, 0)
h19  PDF ISBN 9789211542639   (print ISBN 9789211576092 — different; the trap)
h21  12 orders of magnitude   (body sentence AND Figure 1.6 caption agree)
```

**The body/references split generalizes.** The HDR carries an 87-page reference
section. `metaverse` occurs exactly once in the whole file — inside the title of
a cited ITU report. A whole-document index says "mentioned" (wrong); counting
the body only says "never" (right). **A term appearing only in a cited work's
title is not a subject the document discusses.**

### Parks (`nationalparks_europe.txt`, 93k words — frozen line of work)

- Exactly **8** parks report a summit ≥3000 m.
- **Paklenica's entry never contains the word "Croatia"** — why d03 counts 2.
- `poaching 0 | natura 0 | glaciation 0 | wartime 0 | glacier 63 | glaci* 173`
- `glaciation` 0 but `glacier` 63 → **word families are mandatory.**
- `accessible` 39 (travel sense) but `wheelchair` 0 → **head words mislead.**
- Box 26 Jotunheimen is `1151 Area covered (sq miles)` — the **only** park not
  in sq km. Any pipeline that does not convert units gets the area questions
  wrong in a way that looks plausible.
- Box 24 Hortobágy: area is **820**; 135,000 is cranes. "**Peak** annual
  population" was read as a summit → a 135,000 m peak.
- Box 10 Cairngorms: area 4528, summit **1309**. A *swap* passes any
  plausibility range.

---

## 6. Method — use this before changing anything

**Diagnose against the raw text locally, with no LLM calls, before touching the
pipeline.** Every real fix in this project came from a 20-line script: regexing
all 60 boxes revealed the cranes-as-area mispairing; counting terms proved
`poaching` genuinely absent; grepping `148,063` proved a number was invented;
measuring `chars/4` proved the parks document was only 1.2× the window, which
was the single most valuable measurement of the project.

**Dry-run `__main__` with the LLM mocked before spending a call.** The HDR
pipeline was validated end-to-end this way — 756 calls, every failure path
exercised — for zero broker spend. Mocks are not enough on their own: g1's
`_parse_value` shipped a 3-vs-4 tuple arity bug that no mocked test caught,
because every synthetic record happened to contain a number.

**Read the judge's per-key-point feedback.** It converts "Absence is weak" into
"d09 wants four things and we answer one" — a different problem with a different
fix. Ask for it every run.

**Rules of thumb learned the hard way:**

- Change **one thing** per scored run. v4 changed three and hid a regression.
- Never let a fallback be silent — `decision`, `shortfall` and truncation notes
  exist so a degraded answer is visible in the submission.
- When two components disagree, do not pick by fiat; resolve against data.
- `temperature=0` is not optional. Runs #19 and #21 were the **same code over
  the same data** and scored **67.0% and 60.4%**. The 15 questions backed by a
  computed value scored identically; the 6 free-prose ones moved. **Single-run
  numbers carry ±7 points of noise.** With 3 questions per category, one
  question swings a category by 33 points.
- **Rolling back does not restore a score.** The only artifact that preserves a
  score is the graded file itself.

---

## 7. Compressed history

Parks, `nationalparks_europe.txt`: **38.6% → 82.8%** across 14 versions. Best
single configuration is v12. The four ideas that produced nearly all of the
gain, in order of value:

1. **Anything countable is computed in Python.** The LLM does only what regex
   cannot. (v3: aggregation 0 → 47)
2. **Read the document verbatim in slices sized to the window.** We had been
   *compressing* the document into a 90k-token store and destroying the
   sentence-level facts the judge grades on. (v10 → v11: 62.3% → 80.2%)
3. **Slice size controls whether a reader enumerates or generalizes.** A reader
   holding 19k words writes summaries worth zero key points; one holding ~6k
   words lists named instances. Size by **words per slice**, not slice count.
   (v13)
4. **Union is safe only when every draft is a grounded quote.** v12's "preserve
   every reader's detail" rule was right with 5 readers; with 15 it faithfully
   preserved a hallucinated park name that displaced a correct computed value.
   The fix is to **rank sources, not pool them**: computed values in an
   un-overrulable block, text evidence unioned below. (v14 — the most
   transferable prompt-level lesson in the project)

HDR line: `h1` (v14's answering half + fitted compiler) → `h2` (stopped
laundering computed values through the model; 9/21 answered in Python with zero
LLM calls; 17.3 min wall clock) → **`g1` (this document's subject: the compiler
deleted, everything discovered at runtime)**.

The `h2` run took **17.3 minutes** for 432 calls (~25 calls/min). g1 makes ~648
calls because aggregation and superlative go through readers instead of a fitted
compiler, so budget **25–35 minutes**. Both fit inside the 60-minute window.

---

## 8. Team and logistics

Four-person team, DASU/AICOSS Summer 2026 Capstone, "Large context Beyond RAG".
Two pipelines are being compared head-to-head on documents larger than the
original parks file, following the protocols in `md_files/`.

- **One broker key for the whole group.** Serialize runs or the timing and
  quality numbers are both meaningless.
- Validation protocols live in `md_files/` (3 protocols + the parks answer key).
  Protocol 2 freezes an answer key **before** scoring; Protocol 3 forbids
  amending it retroactively.
- Repo: `pyoseonchoi/THU-Capstone`. **Work on a branch, never on `main`.**

---

## 9. 한국어 요약

### 목표

**어떤 크기의 `.txt` 문서라도 읽고**, 주어진 21문항(7개 카테고리 × 3)에
**RAG 없이** 답하는 파이프라인 하나. 대회 당일 문서는 마감 1시간 전에 공개되고,
평가 실행 기회는 팀당 **한 번**입니다. PDF 변환은 필요 없습니다 — `.txt`와 질문
파일이 주어집니다.

**특정 문서 전용 모델이 아닙니다.** 코드 안의 상수가 특정 문서를 가리키면
그것은 최적화가 아니라 버그입니다.

### 이미 두 번 실패한 방식: 문서 전용 컴파일러

`pipeline_v3`+는 `"Park in numbers"` 상자에, `hdr_compile.py`는 288–292쪽에
고정돼 있습니다. 각자 자기 문서에서는 높은 점수를 냈고, 다른 문서에서는 아무것도
찾지 못합니다. 특히 위험했던 것: absence 문항이 **우리 변환기가 써넣은**
`[page N]` 마커에 의존하고 있었습니다. 주최측이 준 `.txt`에는 마커가 없으므로
본문이 빈 문자열이 되고, 54번 나오는 단어를 "한 번도 언급되지 않았다"고
자신 있게 답합니다 — 채점 공식에서 가장 나쁜 결과입니다.

**원칙: 구조는 실행 중에 발견하되, 절대 가정하지 않는다.**

### 현재 결과 (`pipeline_g1.py`, HDR 2025)

**quality = 0.5635 (11.83 / 21)**

- **absence 1.00 — 모델 호출 0회.** 튜닝한 적 없는 문서에서 만점.
- **aggregation 0.00, superlative 0.00.** 0점 7문항이 전부 "표에서 특정 값
  뽑기" 한 종류입니다. 나머지 14문항 평균은 0.845.

원인 세 가지를 원문 대조로 확인했습니다:
1. **틀린 표** — 질문이 "Statistical Annex Table 1"이라고 범위를 명시하는데
   `tally()`가 이를 완전히 무시합니다.
2. **틀린 열** — Norway 0.995. 0.995는 문서에 8번 나오지만 *다른 표의 다른
   열* 값입니다. 숫자는 실재하고 연결이 틀렸습니다.
3. **지어낸 숫자** — Qatar 148,063. 이 문자열은 문서에 **0번** 나옵니다.
   출처 검증 장치가 파이프라인 어디에도 없습니다.

### 다음 수정 (우선순위)

가장 큰 것은 **출처 검증**입니다. 리더가 원문 줄을 그대로 인용하게 하고,
파이썬이 "이 숫자가 원문에서 이 개체명 근처에 실제로 있는가"를 문자열로
확인한 뒤에만 집계에 넣습니다. 모델도 문서 지식도 필요 없는 순수 문자열
검사이며, Qatar/148,063과 Norway/0.995를 둘 다 걸러냈을 것입니다.

### 가장 값진 방법론

**바꾸기 전에 원문을 직접 뒤져라.** 실제로 고쳐진 것은 전부 LLM 호출 없이
20줄짜리 스크립트로 찾았습니다. **브로커 호출 전에 `__main__`을 모의 실행해라** —
다만 모의 실행만으로는 부족합니다. g1의 `_parse_value`는 튜플 개수가 안 맞는
버그를 안고 배포됐는데, 합성 테스트의 모든 레코드에 우연히 숫자가 들어 있어서
한 번도 걸리지 않았습니다.
