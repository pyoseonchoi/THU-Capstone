# Live Dashboard — Chat Workspace Redesign

Date: 2026-08-06
Author: 손채은 (with Claude)
Branch: `feature/live-dashboard`

## Goal

The dashboard shipped in `2026-08-04-live-dashboard-design.md` works, and this
session added evidence quotes (`evidence_quotes` on `PipelineAnswer`), a
source viewer (PDF.js highlight / TXT `<mark>` highlight, currently a modal),
upload delete buttons, and a Pipeline Parameters cleanup (removed 7 fields
that don't actually affect pipeline behavior: `verifier_model`,
`planner_model`, `chunk_target_tokens`, `chunk_overlap_tokens`,
`llm_temperature`, `llm_top_p`, `llm_seed`).

The layout itself is now the problem: four section tabs (Upload / Run
Questions / Progress / Results) spread related information across screens
that don't reinforce each other, nothing is visually primary, and there's no
way to type a question directly — only picking from an uploaded question
set. This spec replaces the tabbed layout with a single always-visible
3-column workspace, and maps the redesign directly onto DASU's own bonus
criteria (`Project_Info.md`, `DASU_Project_Pitch_Summer_2026.pdf` slides
17/19): **transparency** ("what is the system doing right now", shown with
their own ChatGPT-activity-panel example) and **performance** (mixed-model
routing and/or prompt caching).

`app/` pipeline logic (parsing, compiling, mapping, reducing, answering) is
unchanged. This spec's only backend touch is one additive field in
`_serialize_run`'s usage block (cache-hit summary) — see below.

## Non-goals (YAGNI)

- No new SSE fields, no widening `ProgressCallback`. Same constraint as the
  original spec: that type is shared across `app/pipeline.py` and the
  mapping-stage files another pair is actively changing on `model-update`.
  Live Activity renders from the stage name alone, exactly like the current
  Progress Log does.
- No fabricated "cost if we hadn't routed to a small model" comparison —
  there's no counterfactual run to honestly back that number. Performance
  panel shows real, defensible numbers only: actual per-model token/cost
  split (already computed, currently drawn as a donut) and real cache-hit
  rate (computed from `UsageRecord.cached`, not previously surfaced).
- No chat history persistence across page reload — same in-memory-only
  model as today; `GET /runs/{run_id}` remains the durable fallback.
- No multi-document workspace — one active document at a time, same as today.
- No change to the `evidence_quotes` / `source_pages` API contract shipped
  earlier this session.

## Architecture

```
web/
├── index.html   3-column grid replaces .app's current single-column flow
├── styles.css   new grid + chat-bubble + docked-viewer rules; section-tab
│                 rules deleted (component removed, not just hidden)
└── app.js       same file, reorganized into three groups of functions
                 (Document panel / Chat panel / Performance panel) sharing
                 one `state` object, as today
```

Column split: 25% / 50% / 25% (`grid-template-columns: 1fr 2fr 1fr`), single
row, each column independently scrollable. No section tabs, no modal for the
source viewer — the viewer is permanently docked in the left column.

**Left — Document panel**
- Upload dropzones (unchanged component, existing delete buttons)
- Uploaded file list (unchanged)
- Docked source viewer: PDF.js canvas + highlight layer, or `<pre>` with
  `<mark>` for TXT — the same rendering code shipped this session, just
  mounted permanently instead of inside `#source-modal`. Empty state: "Upload
  a document to preview it here."

**Center — Chat panel**
- Question picker (existing 10-row listbox, for a loaded question set)
- New: free-text input + "Ask" button, calling the same `startRun()` path
  with a single `{question}` object — no `id`/`category` required, since
  `QuestionRequest.question_id` already auto-generates and
  `question_compiler.py`'s routing already falls back to keyword matching
  (verified: `category` is a hint, not a requirement) when category is "".
- Chat thread (replaces both the old single "answer" card and the Question
  History table, per approval): each run appends one turn — a user bubble
  (the question) followed by an assistant bubble that starts in a "thinking"
  state (Live Activity text + model badge, sourced from the same SSE stream
  as today) and finalizes into the answer text + evidence chips. Evidence
  chip's "View in source" no longer opens a modal — it calls
  `DocumentViewer.showQuote(quote, page)` directly.
- Failed runs render as an error-styled assistant bubble in place, instead of
  the current global error banner (upload-time errors keep the banner —
  those aren't a chat turn).

**Right — Performance panel**
- Compressed stat row (cost, tokens — same data as today's top stats)
- New cache-hit-rate stat + per-model token/cost split (same numbers the
  donut shows today, restyled as a stat block instead of a chart, since a
  donut and a chat thread's model badges were showing overlapping
  information)
- Pipeline Parameters, collapsed under a disclosure at the bottom (same 9
  fields already cleaned up this session — no further changes)

## Backend changes (additive only)

`run.usage` (`list[UsageRecord]`, each with `.cached: bool`, `.model`,
`.input_tokens`, `.output_tokens`) is already computed and saved to
`data/runs/<run_id>.json`, but `_serialize_run` in `app/api/main.py` never
surfaces it beyond `total_input_tokens`/`total_output_tokens`. Add one
computed block:

```python
"usage": {
    "total_input_tokens": run.total_input_tokens,
    "total_output_tokens": run.total_output_tokens,
    "cache": {
        "cached_calls": <count where .cached>,
        "total_calls": len(run.usage),
        "hit_rate": <cached_calls / total_calls, or None if total_calls == 0>,
    },
    "by_model": {model: {"input_tokens": ..., "output_tokens": ...}, ...},
},
```

`app.js` accumulates `cache.cached_calls`/`cache.total_calls` and `by_model`
across every completed run in the session (same accumulation pattern the
top stat cards already use) to drive the Performance panel — no new
endpoint, no new polling interval.

**Open item to confirm during implementation**: whether a cache-hit
`UsageRecord` logs the token count the skipped call *would have* used, or
`0`/`None` since no request was actually sent. This changes whether the
Performance panel can also claim "tokens saved by cache" or only a hit-rate
percentage. Check `app/v3/exhaustive_mapper.py`'s cache-write path before
implementing that specific number; hit-rate itself doesn't depend on the
answer.

## Data flow

1. Upload document (left) → `POST /documents` → shown in file list, viewer
   shows an empty/first-page state.
2. Upload question set (left, optional) → picker populated (center) — or
   skip this and type a question directly (center).
3. Submit (pick + "Run this question", or type + "Ask") → chat thread (center)
   appends a user bubble immediately, `POST /answer-jobs` → `job_id`.
4. `GET /answer-jobs/{job_id}/stream` drives that turn's assistant bubble:
   stage text + model badge update live ("Thinking...", "Mapping evidence —
   34/63 records", ...), exactly the event shape already implemented.
5. `done` → assistant bubble finalizes with the answer + evidence chips;
   completed run's `usage` block folds into the Performance panel's running
   totals.
6. Clicking a chip's "View in source" → left panel's viewer jumps to that
   page/quote and highlights it (`DocumentViewer.showQuote`), no modal open/close.
7. Right panel's cost/token stats also keep refreshing independently via the
   existing 5s `/usage` poll, unchanged.

## Error handling

- Upload failure → inline banner under the dropzone (unchanged).
- Run failure → that turn's assistant bubble renders in an error state
  in-place (message text + retry affordance), instead of a page-level banner.
- SSE drop → unchanged: `EventSource` auto-retries; if it gives up, the turn
  shows "Reconnect" — the job survives server-side and `GET
  /answer-jobs/{job_id}` can still be polled directly.
- Viewer fetch failure (`/documents/{id}/file` 404, e.g. server restarted and
  the upload directory changed) → inline message inside the left panel only;
  does not block the chat thread.

## Testing

- Backend: extend `tests/test_api.py` to assert `_serialize_run`'s new
  `usage.cache` block has correct `hit_rate` arithmetic (including the
  `total_calls == 0` edge case) against a constructed `PipelineRun` fixture.
- Frontend: no test framework, per existing convention. Manually verify
  against mocked `fetch`/`EventSource`: empty state (no document), thinking
  state (turn in progress), completed turn with multi-page evidence (viewer
  jumps correctly across two different quotes' pages), and a failed turn.

## Open items carried forward, not blocking

- Exact cache-hit token semantics (see Backend changes above).
- Whether `Pipeline Parameters` stays collapsed-by-default or expanded —
  low-stakes, decide visually during implementation.
