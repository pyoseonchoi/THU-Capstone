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
`_serialize_run`'s usage block (per-model token split) — see below.

**Cache-hit rate is explicitly out of scope for this pass** (see Non-goals):
review of the existing code found that a cache hit in
`app/v3/exhaustive_mapper.py` returns early without ever calling
`self._tracker.record(...)`, so no `UsageRecord` is created for it — there is
currently no data to compute a hit rate from. Making that number real would
require adding a tracker call on the cache-hit path, which touches a
mapping-stage file another pair is actively changing on `model-update`.
Deferred to a follow-up spec once that coordination happens.

## Non-goals (YAGNI)

- No new SSE fields, no widening `ProgressCallback`. Same constraint as the
  original spec: that type is shared across `app/pipeline.py` and the
  mapping-stage files another pair is actively changing on `model-update`.
  Live Activity renders from the stage name alone, exactly like the current
  Progress Log does.
- No fabricated "cost if we hadn't routed to a small model" comparison —
  there's no counterfactual run to honestly back that number. Performance
  panel shows real, defensible numbers only: actual per-model token/cost
  split (already computed per-call, just not currently surfaced by
  `_serialize_run`).
- No cache-hit-rate stat this pass — see Goal section for why (the
  underlying `UsageRecord.cached` is never actually set today).
- No resizable columns this pass — fixed 25/50/25 split. Drag-to-resize
  dividers are a real, feasible follow-up (plain pointer-event drag updating
  `grid-template-columns`, no library needed) but are deferred so this spec
  stays focused on the layout/content change; noted under Open items.
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
│                 rules deleted (component removed, not just hidden), and
│                 likewise the entire `/* ---------- source viewer modal
│                 ---------- */` comment block deleted (all of it — header,
│                 title, sub, body, loading, backdrop, panel, close-button
│                 rules, not just the four most obvious ones) — the
│                 `#source-modal` markup and its close-button/backdrop-click/
│                 Escape-key handlers in app.js go with it, since docking the
│                 viewer removes anything to open or close
└── app.js       same file, reorganized into three groups of functions
                 (Document panel / Chat panel / Performance panel) sharing
                 one `state` object, as today
```

Column split: fixed 25% / 50% / 25% (`grid-template-columns: 1fr 2fr 1fr`),
not resizable this pass (see Non-goals), single row, each column
independently scrollable. No section tabs, no modal for the source viewer —
the viewer is permanently docked in the left column.

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
  `openSourceViewer(quote, page)` directly — but that function needs to be
  refactored first: today it's wired to the modal's own show/hide lifecycle
  (toggles `#source-modal`'s `classList`, plus backdrop-click/close-button/
  Escape handlers that open and close it). Docking it permanently means
  splitting out the render logic (PDF.js page render + highlight, or `<pre>`
  + `<mark>` for TXT) from that open/close chrome, since there's no longer
  anything to open or close.
- Failed runs render as an error-styled assistant bubble in place, instead of
  the current global error banner (upload-time errors keep the banner —
  those aren't a chat turn).

**Right — Performance panel**
- Compressed stat row (cost, tokens — same data as today's top stats)
- New per-model token/cost split, as a stat block instead of the current
  donut. This is a deliberate replacement, not just a restyle: the existing
  donut (`updateDonut` in `web/app.js`) tallies SSE progress-*event* counts
  per model badge, not actual tokens/cost — the chat thread's per-turn model
  badges already cover that "which model handled this" signal, so the
  donut's information is redundant once the thread exists. The new stat
  block uses real per-call token data via the `by_model` field below.
- Pipeline Parameters, collapsed under a disclosure at the bottom (same 9
  fields already cleaned up this session — no further changes)

## Backend changes (additive only)

`run.usage` (`list[UsageRecord]`, each with `.model`, `.input_tokens`,
`.output_tokens` — real records, written on every actual LLM call; the
cache-hit gap discussed above only affects a `.cached` flag this spec
doesn't use) is already computed and saved to `data/runs/<run_id>.json`, but
`_serialize_run` in `app/api/main.py` never surfaces it beyond
`total_input_tokens`/`total_output_tokens`. Add one computed block:

```python
"usage": {
    "total_input_tokens": run.total_input_tokens,
    "total_output_tokens": run.total_output_tokens,
    "by_model": {model: {"input_tokens": ..., "output_tokens": ...}, ...},
},
```

Guard the summation against `None` — `UsageRecord.input_tokens`/
`output_tokens` are `Optional[int] = None` (treat as 0 when summing).

`app.js` accumulates `by_model` across every completed run in the session
(same accumulation pattern the top stat cards already use) to drive the
Performance panel — no new endpoint, no new polling interval.

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
   page/quote and highlights it (`openSourceViewer`, refactored to render into
   the docked panel instead of a modal), no open/close transition.
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
  `usage.by_model` block sums tokens correctly per model, including a
  `PipelineRun` fixture with at least one `UsageRecord` whose
  `input_tokens`/`output_tokens` is `None`.
- Frontend: no test framework, per existing convention. Manually verify
  against mocked `fetch`/`EventSource`: empty state (no document), thinking
  state (turn in progress), completed turn with multi-page evidence (viewer
  jumps correctly across two different quotes' pages), and a failed turn.

## Open items carried forward, not blocking

- **Cache-hit rate** (dropped from this pass, see Goal): needs a
  `self._tracker.record(...)` call added on the cache-hit path in
  `app/v3/exhaustive_mapper.py`. Coordinate with whoever owns that file on
  `model-update` before attempting; a follow-up spec once that lands.
- **Resizable columns** (dropped from this pass, see Non-goals): drag
  handles between columns, updating `grid-template-columns` on
  pointermove, clamped to sane min-widths per column. No backend
  involvement; can be added independently whenever.
- Whether `Pipeline Parameters` stays collapsed-by-default or expanded —
  low-stakes, decide visually during implementation.
