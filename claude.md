# Claude Code Handoff - FULLSCAN-QA Pipeline Work

Last updated: 2026-08-05 KST

## Primary Workspace

Continue work in:

```text
C:\Users\w0719\OneDrive\바탕 화면\캡스톤팀\THU-Capstone-ui
```

Do **not** continue in the original teammate folder unless the user explicitly asks:

```text
C:\Users\w0719\OneDrive\바탕 화면\캡스톤팀\THU-Capstone
```

The original teammate repo was intentionally restored/kept clean. The active work is in `THU-Capstone-ui`.

## Current Git State

Branch:

```text
codex/live-dashboard-work
```

Known working-tree changes in `THU-Capstone-ui`:

```text
 M app/config.py
 M app/pipeline.py
 M app/v3/answerer.py
 M app/v3/compiler.py
 M app/v3/models.py
 M app/v3/question_compiler.py
 M app/v3/structured_executor.py
?? app/v3/table_compiler.py
?? samples/
?? scripts/run_eval_set.py
```

Nothing has been committed or pushed yet.

## Project Rules To Preserve

From `AGENTS.md`:

- No RAG, vector DB, embeddings, BM25, TF-IDF, top-k retrieval, reranking, or query-dependent filtering.
- The system must scan/compile the whole document.
- LLMs may extract or polish evidence-grounded answers, but deterministic operations must remain in Python.
- Python must compute counts, maxima, minima, sums, averages, differences, absence logic, comparisons, etc.
- Do not hardcode API keys. Use environment variables only.
- Do not commit `.env` files.
- Every chunk should have a terminal status if it reaches the mapping path.

## What Was Implemented

The pipeline was strengthened for large full-document QA datasets, especially OECD/HDR-style reports.

Main implementation areas:

1. `app/v3/models.py`
   - Added/expanded structured model fields for operation plans, table rows, contents entries, and compiled document metadata.
   - `CompiledDocument` now carries contents/table/registry signals used by deterministic executors.

2. `app/v3/compiler.py`
   - Compiler version advanced to v3.4.
   - Improved compilation of long reports, contents, repeated records, inline facts, number facts, country/title signals, and document-level registry data.

3. `app/v3/table_compiler.py`
   - New file.
   - Handles contents/table-style extraction used by report datasets.

4. `app/v3/question_compiler.py`
   - Adds richer operation planning.
   - Handles absence options and composable question operations.

5. `app/v3/structured_executor.py`
   - Adds deterministic Python answer paths for OECD/HDR/report-style questions.
   - Includes deterministic answerers for aggregation, table/contents questions, absence questions, contradictions, cross-section joins, global synthesis, and needles.

6. `scripts/run_eval_set.py`
   - New evaluation runner.
   - Supports:
     - `--offline-deterministic`
     - `--llm-finalize-deterministic`
   - Saves `submission.json`, `run_result.json`, `summary.json`, and appends run metadata.

7. `app/config.py`, `app/pipeline.py`, `app/v3/answerer.py`
   - Added optional deterministic-answer finalization with an LLM so token/time metrics are recorded while Python still computes the answer.

## Validation Already Run

Regression tests:

```powershell
pytest tests -q -m "not live" -p no:cacheprovider
```

Result previously observed:

```text
102 passed, 1 warning
```

Lint/format check previously observed:

```powershell
ruff check app/v3 app/config.py app/pipeline.py scripts/run_eval_set.py --no-cache
```

Result previously observed:

```text
All checks passed
```

## Dataset Folder

User dataset folder:

```text
C:\Users\w0719\OneDrive\바탕 화면\데이터셋
```

Files currently known:

```text
grand_river_projects_large.txt
grand_river_questions.json
hdr2025_answer_key.yaml
hdr2025_fullscan_questions.json
hdr2025_fullscan_test.txt
meridian_field_stations_v2.txt
meridian_questions_v2.json
nationalparks_europe.txt
oecd2026_fullscan_questions.json
oecd2026_fullscan_test.txt
questions.json
```

## OECD API Run Already Completed

The user ran this API-backed deterministic-finalization evaluation:

```powershell
cd "C:\Users\w0719\OneDrive\바탕 화면\캡스톤팀\THU-Capstone-ui"

$env:LLM_PROVIDER="openai_compatible"
$env:LLM_BASE_URL="http://80.151.131.52:9839/v1"
$env:MISTRAL_API_KEY="<do not hardcode; user-provided secret>"
$env:ANSWER_MODEL="ministral-3b-2512"

python -B scripts/run_eval_set.py `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_test.txt" `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_questions.json" `
  --name oecd_api_test `
  --llm-finalize-deterministic
```

Output directory:

```text
data\eval_runs\oecd_api_test\20260805_202132
```

Submission:

```text
data\eval_runs\oecd_api_test\20260805_202132\submission.json
```

Summary:

```text
questions:     21
chunks:        93
LLM calls:     21
input tokens:  13,896
output tokens: 3,155
total tokens:  17,051
wall seconds:  31.91
```

## Self-Scoring Status

The user provided three protocols:

```text
C:\Users\w0719\Downloads\1_question_generation_protocol.md
C:\Users\w0719\Downloads\2_answer_key_protocol.md
C:\Users\w0719\Downloads\3_scoring_protocol.md
```

Important Protocol 3 rule:

- Formal reproducible scoring requires a frozen `answer_key.yaml`.
- If no answer key exists, Protocol 2 says to build/freeze one first.

For OECD, no frozen `oecd2026_answer_key.yaml` was found locally. A manual/provisional Protocol-2-style key was inferred from the source and questions, then used to self-score the submitted answers.

Provisional self-score for OECD run:

```text
quality = 1.00
raw score = 21.0 / 21
```

Category breakdown was effectively 1.00 for every category:

```text
aggregation      1.00
superlative      1.00
absence          1.00
contradiction    1.00
cross_section    1.00
global_synthesis 1.00
needle           1.00
```

## Important Risk Found

`q11` in the OECD API run contains the correct factual content, but the final answer string is malformed because the LLM finalization returned a broken JSON-like nested string.

Observed answer:

```text
{"answer": {"answer": {"Sweden in Table 5.1: 409 surveyed companies; Sweden in Annex Table 5.A.5: 408 surveyed companies."
```

The content contains the needed key points:

- Sweden
- Table 5.1 = 409 surveyed companies
- Annex Table 5.A.5 = 408 surveyed companies

So manual Protocol 3 scoring gives full credit. However, the formatting is risky for strict graders or human review.

## Recommended Next Work

Priority 1: Make deterministic LLM finalization safe.

In `app/v3/answerer.py`, improve `finalize_deterministic()` so it keeps the original Python-computed answer when the LLM output is malformed or structurally suspicious.

Suggested behavior:

- Call the LLM to record token/time as now.
- Parse response.
- Accept the LLM-polished answer only if it is a clean non-empty natural-language/string answer.
- Reject/fallback if the parsed answer:
  - looks like broken JSON,
  - has unbalanced braces/brackets,
  - begins with `{` or `[`,
  - is much shorter than the deterministic original while dropping key numbers/entities,
  - or otherwise fails a simple sanity check.
- When rejected, return the deterministic Python answer and add a warning like:

```text
LLM finalization rejected; retained deterministic Python answer.
```

This preserves the design: Python computes, LLM call still records token/time, and the final answer cannot be degraded by formatting failures.

Priority 2: Rerun OECD API test after the guard.

Use the same command as above, but with a new name:

```powershell
python -B scripts/run_eval_set.py `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_test.txt" `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_questions.json" `
  --name oecd_api_guarded `
  --llm-finalize-deterministic
```

Then inspect:

```text
data\eval_runs\oecd_api_guarded\<timestamp>\submission.json
data\eval_runs\oecd_api_guarded\<timestamp>\summary.json
```

Priority 3: Generate/freeze answer keys for reproducible scoring.

Known existing frozen key:

```text
C:\Users\w0719\OneDrive\바탕 화면\데이터셋\hdr2025_answer_key.yaml
```

Missing but useful:

```text
oecd2026_answer_key.yaml
nationalparks_answer_key.yaml
grand_river_answer_key.yaml
meridian_answer_key.yaml
```

If building a scorer, follow `3_scoring_protocol.md` exactly:

```text
question_score = max(0, covered / total_key_points - 0.25 * forbidden_hits)
quality = mean(question_score over all 21 questions)
```

Priority 4: Run all five datasets and record token/time/score.

Recommended table columns for the final report:

```text
dataset
questions
chunks
llm_calls
input_tokens
output_tokens
total_tokens
wall_seconds
self_score_or_official_score
notes
```

## Quick Commands

Offline deterministic smoke test:

```powershell
python -B scripts/run_eval_set.py `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_test.txt" `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_questions.json" `
  --name oecd_offline_check `
  --offline-deterministic
```

API-backed token/time run:

```powershell
$env:LLM_PROVIDER="openai_compatible"
$env:LLM_BASE_URL="http://80.151.131.52:9839/v1"
$env:MISTRAL_API_KEY="<secret>"
$env:ANSWER_MODEL="ministral-3b-2512"

python -B scripts/run_eval_set.py `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_test.txt" `
  "C:\Users\w0719\OneDrive\바탕 화면\데이터셋\oecd2026_fullscan_questions.json" `
  --name oecd_api_guarded `
  --llm-finalize-deterministic
```

Regression tests:

```powershell
pytest tests -q -m "not live" -p no:cacheprovider
```

Lint:

```powershell
ruff check app/ tests/ scripts/ --no-cache
```

## Notes For Claude

- Be careful with Korean Windows paths in Python scripts. Passing absolute Korean paths through some PowerShell/Python paths has produced mojibake in output. Running from the repo and using relative paths for generated artifacts is safer.
- Do not print or save the user's API key.
- Do not replace deterministic reducers with LLM logic.
- The immediate practical bug to fix is the malformed q11 LLM finalization output.
- After the guard is implemented, rerun OECD and verify `q11` is a clean answer.
