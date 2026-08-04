# Live Dashboard Frontend — Design

Date: 2026-08-04
Author: 손채은 (frontend/backend pair, with Claude)
Branch: `feature/live-dashboard`

## Goal

`app/` already implements the team's FULLSCAN-QA v3 pipeline (deterministic
document compiler, exhaustive mapper, category-specific reducers, FastAPI
endpoints) behind a Streamlit demo UI. This spec adds a second, purpose-built
frontend for live demoing and day-of-competition operation: document upload,
question-set upload, one-click full run, a real-time progress log with
model-routing badges, a running cost counter, and a `submission.json`
download — without changing pipeline behavior.

`app/` is out of scope except for the small, additive changes listed below.
No changes to chunking, compiling, mapping, reduction, or verification logic.

## Non-goals (YAGNI)

- No database. Question/run history lives in browser memory for the session;
  the server's `/runs` store is the source of truth if the tab is lost.
- No YAML question-file support. The practice `dev_questions.yaml` must be
  converted to JSON (`{id, question, category}` list) before upload — the
  existing `parse_questions_json` on the backend is the only parser touched.
- No auth, no multi-user session handling — single group key, single browser
  tab, matches how the competition is actually run.
- No build tooling (no npm/webpack/vite). Plain HTML/CSS/JS, optionally
  Alpine.js via CDN if hand-written state sync gets unwieldy.

## Architecture

```
THU-Capstone/
├── app/                  existing pipeline + API — untouched except below
│   └── api/main.py       + CORS middleware, + SSE route, + job queue
├── ui/streamlit_app.py   existing demo — left as-is, not removed
├── web/                  NEW — static dashboard frontend
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── docs/superpowers/specs/   this file
```

`web/` is served independently (e.g. `python -m http.server` from inside
`web/`) and talks to the FastAPI backend on `http://127.0.0.1:8000` over
`fetch`/`EventSource`. Kept separate from `app/` so the pipeline team's
work and the frontend team's work don't collide on the same files, beyond
the one shared touch point (`app/api/main.py`).

## Backend changes (additive only)

In `app/api/main.py`:

1. Add `CORSMiddleware` (allow the `web/` static origin during local dev).
2. Add a per-job `asyncio.Queue` alongside the existing `_answer_jobs[job_id]`
   dict, created in `create_answer_job` and torn down when the job finishes.
3. Extend the `progress` closure inside `_execute_answer_job` — which
   already receives `(stage, processed, total, failed)` from the existing
   `ProgressCallback` — so that, in addition to updating
   `_answer_jobs[job_id]`, it also pushes `{ts, stage, processed, total,
   failed}` onto that job's queue. No new data is needed from the callback
   itself.
4. New route: `GET /answer-jobs/{job_id}/stream` — a `StreamingResponse`
   that yields `data: {json}\n\n` for each queued event, then a final
   `event: done\ndata: {}\n\n` and closes when the job's status becomes
   `completed` or `failed`.

**Model badge is derived client-side, not sent by the backend.**
`ProgressCallback` is `Callable[[str, int, int, int], None]` — a single type
alias shared verbatim across `app/pipeline.py`, `app/mapping/batch_mapper.py`,
and `app/v3/exhaustive_mapper.py` (6+ call sites). Widening it to also carry
a model name would mean editing the mapping-stage files, which directly
contradicts this spec's own boundary ("no changes to ... mapping ...
logic") and risks colliding with the pipeline pair's active work.

Instead: models are already assigned statically per stage via config
(`mapper_model`, `answer_model`, ...; `batch_mapper.repair_failures` even
uses `model_override=answer_model` for its retries, so stage already implies
model). `web/app.js` keeps a small static lookup —
`{compiling: null, mapping: "ministral-3b-2512", repairing: "ministral-3b-2512" /* or answer_model, see below */, answering: "mistral-small-3-2"}`
— and renders the badge from the incoming `stage` field alone. Zero touches
to `app/pipeline.py` or the mapping files. (Confirm the `repairing` stage's
actual model with the pipeline pair before hardcoding it — `repair_failures`
was observed using `answer_model` for at least some retries — but that's a
one-line lookup-table fix either way, not an API contract change.)

## Frontend (`web/`)

Visual design: the dashboard mockup already approved (sidebar nav, stat
cards row, document-upload card, ask/answer card, progress-log card with
pill filters, model-mix donut, history table) — ported from inline
`<style>` into `styles.css`, IDs/classes added for JS hooks, dummy content
replaced with live bindings.

`app.js` responsibilities (plain functions + a small state object, no
framework):

- `uploadDocument(file)` → `POST /documents` → store `document_id`,
  `chunk_count`; update the upload card.
- `loadQuestions(file)` → parse the uploaded JSON client-side into
  `{id, question, category}[]`; render the count, no hardcoded assumptions
  about how many.
- `startRun(documentId, questions)` → `POST /answer-jobs` → `job_id`.
- `streamProgress(jobId)` → `new EventSource('/answer-jobs/' + jobId +
  '/stream')`; each message appends a log row (with model badge if present)
  and updates the stage pill / processed-of-total counter; `done` triggers
  the final render.
- `pollUsage()` → `setInterval` hitting `GET /usage` every 5s, updates the
  top cost counter.
- `renderResult(result)` → fills in the answer card for the active
  question, appends a row to the history table, and keeps `result.submission`
  in memory for download.
- `downloadSubmission()` → serializes the in-memory `submission` object to
  a `Blob` and triggers a save — no extra request needed since `/answer-jobs`
  already returns it in the completed job payload (falls back to
  `GET /runs/{run_id}/submission` if the tab was reloaded mid-run).

## Data flow

1. Upload document → `POST /documents` → `document_id`.
2. Upload question set (JSON) → parsed client-side → question list, any
   length.
3. "전체 실행" → `POST /answer-jobs {document_id, questions}` → `job_id`.
4. Subscribe to `GET /answer-jobs/{job_id}/stream` → log rows stream in,
   stat cards and the 3B/24B donut update as events arrive.
5. `done` event → render final answers + submission, populate history table.
6. Download button → save `submission.json` from memory.
7. Cost counter refreshes independently via 5s `/usage` polling throughout.

## Error handling

- Upload failure (400/500 from `/documents`) → inline error banner under the
  dropzone, upload stays retryable.
- SSE drop → `EventSource` retries automatically; if it gives up, show a
  "새로고침해서 이어보기" notice — safe because the job lives in server
  memory and `GET /answer-jobs/{job_id}` can be re-polled directly.
- Per-question failure is already surfaced by the backend via
  `answers[].warnings`; the history table marks just that row "실패" and the
  rest of the run continues — matches the competition's partial-submission
  rule (unanswered = 0, but a partial file still beats no file).

## Testing

- Backend: extend `tests/test_api.py` with a test that drives
  `_execute_answer_job`'s progress callback against a fake queue and asserts
  the SSE route emits well-formed `data:` lines ending in a `done` event.
- Frontend: no test framework added; verify the four visual states
  (idle / uploading / running / done-with-one-failure) by hand in a browser
  against mocked `fetch`/`EventSource` responses before wiring the real
  backend, matching the LLM-free validation approach used for the pipeline
  notebooks.

## Open items carried forward, not blocking

- Alpine.js (CDN, no build step) as a fallback if hand-rolled state sync in
  `app.js` gets messy — not decided upfront, add only if it's actually
  needed.
- Confirm which model `repairing` actually uses with the pipeline pair
  (observed `batch_mapper.repair_failures` passing `model_override=answer_model`
  for at least some retries) before finalizing the client-side stage→model
  lookup table.
