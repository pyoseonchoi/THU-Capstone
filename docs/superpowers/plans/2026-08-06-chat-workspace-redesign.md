# Chat Workspace Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tabbed `web/` dashboard (Upload / Run Questions / Progress / Results) with a single always-visible 3-column workspace: a left Document panel (upload + docked PDF/TXT viewer), a center Chat panel (question picker + free-text input + a chat-thread of turns), and a right Performance panel (compressed stats + real per-model token split + collapsed Pipeline Parameters).

**Architecture:** Three files change: `app/api/main.py` gets one additive `by_model` field in `_serialize_run`'s usage block (backed by data already recorded, just not surfaced). `web/index.html`, `web/styles.css`, and `web/app.js` are restructured in place — same static-file, no-build-step approach, same backend contract otherwise (SSE stream, `/answer-jobs`, `/documents/{id}/file` all unchanged).

**Tech Stack:** FastAPI (Python), vanilla HTML/CSS/JS (no framework, no build step), PDF.js (CDN) for PDF rendering, `pytest` for the one backend test.

**Spec:** `docs/superpowers/specs/2026-08-06-chat-workspace-redesign-design.md`

---

## Before you start

Both servers this plan is verified against:
- Backend: `cd` into the repo root, `.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8001` (port 8000 is used by another session — check `web/app.js`'s `API_BASE` matches whatever port you actually run on).
- Static frontend: `python -m http.server 8080` from inside `web/`.

There is no frontend test framework in this repo (confirmed convention from the prior spec) — frontend tasks are verified by hand in a browser, not with automated tests. The one automated test in this plan is for the backend change in Task 1.

---

### Task 1: Backend — surface real per-model token usage

**Files:**
- Modify: `app/api/main.py` (the `_serialize_run` function)
- Test: `tests/test_api.py`

- [ ] **Step 1: Read the current `_serialize_run` and an existing test for its shape**

Open `app/api/main.py` and find `_serialize_run`. Open `tests/test_api.py` and find any existing test that builds a `PipelineRun` fixture (to match its construction pattern — field names, how `PipelineAnswer`/`UsageRecord` are constructed in tests).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_api.py` (adjust imports at the top of the file to match what's already imported — you'll need `PipelineRun`, `PipelineAnswer`, `UsageRecord` from `app.schemas`, plus whatever `_serialize_run` needs to import from `app.api.main`):

```python
def test_serialize_run_by_model_usage_split():
    from app.api.main import _serialize_run
    from app.schemas import PipelineRun, PipelineAnswer, UsageRecord

    run = PipelineRun(
        run_id="r1",
        document_id="d1",
        answers=[PipelineAnswer(question_id="q1", final_answer="42")],
        usage=[
            UsageRecord(model="ministral-3b-2512", input_tokens=100, output_tokens=20),
            UsageRecord(model="ministral-3b-2512", input_tokens=50, output_tokens=10),
            UsageRecord(model="mistral-small-3-2", input_tokens=200, output_tokens=None),
        ],
        total_input_tokens=350,
        total_output_tokens=30,
    )

    result = _serialize_run(run)

    by_model = result["usage"]["by_model"]
    assert by_model["ministral-3b-2512"] == {"input_tokens": 150, "output_tokens": 30}
    # None output_tokens must not raise or silently corrupt the sum -- treat as 0.
    assert by_model["mistral-small-3-2"] == {"input_tokens": 200, "output_tokens": 0}


def test_serialize_run_by_model_usage_empty():
    from app.api.main import _serialize_run
    from app.schemas import PipelineRun

    run = PipelineRun(run_id="r2", document_id="d1", answers=[], usage=[])

    result = _serialize_run(run)

    assert result["usage"]["by_model"] == {}
```

Check `PipelineRun`'s required fields in `app/schemas.py` first — if it needs more than `run_id`/`document_id`/`answers`/`usage`/`total_input_tokens`/`total_output_tokens` to construct (e.g. a `pipeline_mode`, `started_at`), add those with reasonable defaults so the fixture actually builds.

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -k by_model_usage -v`
Expected: FAIL — `KeyError: 'by_model'` (the field doesn't exist yet).

- [ ] **Step 4: Implement `by_model` in `_serialize_run`**

In `app/api/main.py`, find the `"usage"` block inside `_serialize_run`:

```python
        "usage": {
            "total_input_tokens": run.total_input_tokens,
            "total_output_tokens": run.total_output_tokens,
        },
```

Replace it with:

```python
        "usage": {
            "total_input_tokens": run.total_input_tokens,
            "total_output_tokens": run.total_output_tokens,
            "by_model": _usage_by_model(run.usage),
        },
```

Above `_serialize_run`, add the helper:

```python
def _usage_by_model(records: list) -> dict:
    """Real per-model token totals from recorded LLM calls (each UsageRecord
    is written on an actual call; input/output tokens are Optional[int], so
    a missing value contributes 0 rather than raising).
    """
    totals: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = totals.setdefault(record.model, {"input_tokens": 0, "output_tokens": 0})
        bucket["input_tokens"] += record.input_tokens or 0
        bucket["output_tokens"] += record.output_tokens or 0
    return totals
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -k by_model_usage -v`
Expected: 2 passed.

- [ ] **Step 6: Run the full backend test suite to check for regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: all passing (same pass count as before this change, plus the 2 new tests).

- [ ] **Step 7: Commit**

```bash
git add app/api/main.py tests/test_api.py
git commit -m "feat(api): surface real per-model token usage in _serialize_run"
```

---

### Task 2: HTML — replace the tabbed shell with the 3-column workspace

**Files:**
- Modify: `web/index.html` (full replacement of the `<body>` content)

This task only restructures markup — every `id` that existing JS already binds to (`doc-dropzone`, `doc-input`, `questions-dropzone`, `questions-input`, `file-list`, `chunk-summary`, `chunk-summary-count`, `question-picker`, `question-picker-summary`, `picked-category`, `run-selected-btn`, `run-all-btn`, `download-submission-btn`, `submission-progress-label`, `stat-cost`, `stat-progress`, `stat-tokens`, `stat-chunks`, `stat-chunks-status`, `stat-question-count`, `settings-grid`, `upload-error`, `source-pdf-prev`, `source-pdf-next`, `source-pdf-page-label`, `source-pdf-canvas`, `source-pdf-highlight-layer`, `source-text-pane`, `status-toggle`, `status-toggle-value`, `status-panel`, `index-doc-label`, `index-bar`, `index-chunk-count`) is preserved so Task 1's HTML change and Tasks 3-5's JS changes don't have to land in lockstep. New ids this task introduces (`viewer-body`, `viewer-empty`, `viewer-sub`, `chat-thread`, `chat-empty`, `freeform-input`, `freeform-ask-btn`, `perf-models`) are consumed by Tasks 4-6.

Removed from this markup: the `<nav class="section-tabs">` block, the `<section id="progress-section">` Progress Log + Model Usage Mix donut cards (replaced by the chat thread's per-turn status and the new Performance panel), the `<section id="results-section">` history table, and the entire `<div class="source-modal" id="source-modal">` block (replaced by a docked viewer in the left column). The floating `status-widget` sidebar (`<aside class="sidebar status-widget">`) is **not** touched — it stays exactly as-is; removing it wasn't part of the approved spec.

- [ ] **Step 1: Replace the file**

Replace `web/index.html` in full with:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long-Context Challenge — Pipeline Dashboard</title>
<link rel="stylesheet" href="styles.css?v=workspace-1">
</head>
<body>
<div class="app">

  <aside class="sidebar status-widget">
    <button class="status-toggle" id="status-toggle" type="button" aria-expanded="false" aria-controls="status-panel">
      <span class="status-toggle-label">Document status</span>
      <span class="status-toggle-value" id="status-toggle-value">0 / 0</span>
    </button>
    <div class="index-card status-panel" id="status-panel" aria-hidden="true">
      <div class="t">Document status</div>
      <div class="d" id="index-doc-label">Upload a document</div>
      <div class="bar"><i id="index-bar"></i></div>
      <div class="stat">Chunks <b id="index-chunk-count">0 / 0</b></div>
    </div>
  </aside>

  <main>
    <div class="topbar">
      <div class="brand body-brand">
        <div class="mark">LC</div>
        <div class="brand-copy">
          <span>LongContext</span>
          <small>Team Ulkeke · 울크크</small>
        </div>
      </div>
      <h1>Pipeline Dashboard</h1>
    </div>

    <div class="error-banner" id="upload-error"></div>

    <div class="workspace-columns">

      <section class="col col-document">
        <div class="card">
          <div class="card-head"><h2>Document &amp; Question Set</h2><span class="sub">Upload</span></div>

          <div class="upload-grid">
            <div class="upload-slot">
              <div class="upload-slot-label">1 · Document</div>
              <div class="dropzone" id="doc-dropzone">
                <div class="icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 4v11M12 15l7-4M12 15l-7-4M5 11V7l7-4 7 4v4"/></svg>
                </div>
                <div class="t">Drag a document here, or click to upload</div>
                <div class="d">.txt · .pdf — up to 32MB</div>
                <label class="browse" for="doc-input">Choose document</label>
                <input type="file" id="doc-input" accept=".txt,.pdf" hidden>
              </div>
            </div>

            <div class="upload-slot">
              <div class="upload-slot-label">2 · Question set</div>
              <div class="dropzone" id="questions-dropzone">
                <div class="icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 17H7a5 5 0 01-1-9.9A6 6 0 0117 6a4.5 4.5 0 011 8.9M12 11v9M9 17l3-3 3 3"/></svg>
                </div>
                <div class="t">Drag a question file here, or click to upload</div>
                <div class="d">.json · .yaml · .yml</div>
                <label class="browse" for="questions-input">Choose question set</label>
                <input type="file" id="questions-input" accept=".json,.yaml,.yml" hidden>
              </div>
              <div class="sub" id="questions-count-label" style="margin-top:8px;display:block;">not uploaded yet</div>
            </div>
          </div>

          <div class="filelist" id="file-list"></div>

          <div class="chunk-summary" id="chunk-summary" style="display:none;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>
            <span>Split into <b id="chunk-summary-count">0</b> chunks · document uploaded</span>
          </div>
        </div>

        <div class="card viewer-card">
          <div class="card-head"><h2>Source</h2><span class="sub" id="viewer-sub"></span></div>
          <div class="viewer-body" id="viewer-body">
            <div class="viewer-empty" id="viewer-empty">Upload a document to preview it here.</div>
            <div class="source-pdf-wrap" id="source-pdf-wrap" style="display:none;">
              <div class="source-pdf-toolbar">
                <button id="source-pdf-prev" type="button" aria-label="Previous page">&lsaquo;</button>
                <span id="source-pdf-page-label" class="mono"></span>
                <button id="source-pdf-next" type="button" aria-label="Next page">&rsaquo;</button>
              </div>
              <div class="source-pdf-canvas-wrap">
                <canvas id="source-pdf-canvas"></canvas>
                <div id="source-pdf-highlight-layer" class="source-pdf-highlight-layer"></div>
              </div>
            </div>
            <pre class="source-text-pane" id="source-text-pane" style="display:none;"></pre>
          </div>
        </div>
      </section>

      <section class="col col-chat">
        <div class="card chat-card">
          <div class="card-head"><h2>Ask a Question</h2></div>

          <div class="question-picker-panel">
            <div class="question-picker-summary" id="question-picker-summary">Upload a question set to browse questions.</div>
            <select id="question-picker" class="question-picker-list" disabled size="1">
              <option>Upload a question set first</option>
            </select>
          </div>
          <div class="row">
            <span class="chip chip-cat" id="picked-category">uncategorized</span>
            <span class="spacer"></span>
            <div class="ask-button-group" aria-label="Question run actions">
              <button class="primary" id="run-selected-btn" disabled>Run this question</button>
              <button class="primary" id="run-all-btn" disabled>Run all</button>
            </div>
          </div>

          <div class="ask-freeform">
            <input type="text" id="freeform-input" placeholder="Or type your own question…" disabled>
            <button class="primary" id="freeform-ask-btn" disabled>Ask</button>
          </div>

          <div class="chat-thread" id="chat-thread">
            <div class="chat-empty" id="chat-empty">Pick a question above, or type your own, to get started.</div>
          </div>
        </div>
      </section>

      <section class="col col-performance">
        <div class="stats stats-compact">
          <div class="stat-card dark">
            <div class="l">Total cost</div>
            <div class="n" id="stat-cost">—</div>
            <div class="delta"><span class="n2" id="stat-progress">0/0</span>&nbsp;questions done</div>
          </div>
          <div class="stat-card">
            <div class="l">Tokens (in+out)</div>
            <div class="n" id="stat-tokens">—</div>
            <div class="delta">latest run</div>
          </div>
          <div class="stat-card">
            <div class="l">Indexed chunks</div>
            <div class="n" id="stat-chunks">0</div>
            <div class="delta" id="stat-chunks-status">waiting for upload</div>
          </div>
          <div class="stat-card">
            <div class="l">Question set</div>
            <div class="n" id="stat-question-count">0</div>
            <div class="delta">questions loaded</div>
          </div>
        </div>

        <div class="card">
          <div class="card-head"><h2>Performance</h2><span class="sub">this session</span></div>
          <div class="perf-models" id="perf-models">
            <div class="perf-empty">Run a question to see the model breakdown.</div>
          </div>
        </div>

        <div class="card submission-card">
          <div class="card-head">
            <h2>Submission</h2>
            <span class="sub" id="submission-progress-label"></span>
          </div>
          <button class="primary" id="download-submission-btn" disabled style="width:100%;">Download submission.json</button>
        </div>

        <details class="card params-disclosure">
          <summary>Pipeline Parameters <span class="sub">read-only · from .env</span></summary>
          <div class="stats" id="settings-grid" style="margin-top:14px;margin-bottom:0;">
            <div class="stat-card"><div class="l">Loading…</div></div>
          </div>
        </details>
      </section>

    </div>
  </main>
</div>
<script src="https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js"></script>
<script src="app.js?v=workspace-1"></script>
</body>
</html>
```

- [ ] **Step 2: Sanity-check it's well-formed**

Run: `python -c "import xml.dom.minidom, re, pathlib; s = pathlib.Path('web/index.html').read_text(encoding='utf-8'); print('read ok', len(s), 'chars')"`
(This just confirms the file saved correctly with no encoding issues — HTML5 void/optional tags mean a strict XML parse isn't meaningful here, so don't reach for an XML validator on it.)

Open `http://127.0.0.1:8080` in a browser at this point — expect a broken/unstyled 3-column skeleton with non-functional buttons (styles and JS land in Tasks 3-5). That's expected; don't debug JS errors yet.

- [ ] **Step 3: Commit**

```bash
git add web/index.html
git commit -m "feat(web): replace tabbed shell with a 3-column workspace layout"
```

---

### Task 3: CSS — workspace grid, chat bubbles, docked viewer; delete dead rules

**Files:**
- Modify: `web/styles.css`

- [ ] **Step 1: Delete the rules for components Task 2 removed from the HTML**

Delete these blocks entirely (search for each selector; they're contiguous rules currently between lines ~151-232 and ~430-537 per the pre-redesign file):

- `.section-tabs`, `.section-tab-indicator`, `.section-tab`, `.section-tab.on`, `.section-panel`, `.section-stack` (and the `.section-stack .row2` line), and the `.section-tabs{width:100%;}` / `.section-tab{flex:1 0 auto;text-align:center;}` lines inside the `@media (max-width:760px)` block.
- `.row2` — but **not** the whole media query it lives in. The real file has
  `@media (max-width:980px){.row2{grid-template-columns:1fr;} .stats{grid-template-columns:repeat(2,1fr);}}`
  on one line — delete only the `.row2{grid-template-columns:1fr;}` part.
  Keep `.stats{grid-template-columns:repeat(2,1fr);}`: the plain `.stats`
  class (not `.stats-compact`) is still used by Task 2's `#settings-grid`
  markup, and without this rule Pipeline Parameters renders as an
  uncollapsed 4-column grid on narrow viewports once `.workspace-columns`
  stacks to one column below 1100px.
- `.donut-wrap`, `.donut`, `.donut::after`, `.donut-legend`, `.legend-row`, `.swatch`, `.legend-row .n`.
- `.log`, `.log li`, `.log li:last-child`, `.dot`, `.dot.pending`, `.dot.active`, the `@media (prefers-reduced-motion: reduce){.dot.active{...}}` rule, `@keyframes pulse`, `.log .step`, `.log .step.pending`, `.log .meta`, `.log .time`.
- `.pillnav`, `.pillnav button`, `.pillnav button.on`.
- `.history`, `.table-wrap`, `table`, `thead th`, `tbody td`, `tbody tr:last-child td`, `.qid`, `.qtext`, `.status-pill`, `.status-pill.ok`.
- `tbody tr.history-row`, `tbody tr.history-row:hover td`, `tr.history-detail td`, `.history-detail-inner`, `.history-detail-coverage`, `.history-detail-coverage b`.
- `.ask-hero`, `.ask-hero .card-head`, `.ask-hero .card-head h2`, `.ask-hero select`, `.ask-hero .row`, `.ask-hero button.primary`, `.ask-hero .answer .t`, `.ask-hero .answer-meta`, `.ask-hero .answer` (the whole `.ask-hero` family — it assumed a wide, centered hero card; the new `.chat-card` in a 50%-width column needs different rules, added in Step 3 below).
- From the `@media (max-width:760px)` block, also drop the two `.ask-button-group` rules only if you're keeping the button group's existing non-media styling as-is (see Step 4) — otherwise leave them.
- From `/* ---------- source viewer modal ---------- */`: delete `.source-modal`, `.source-modal.open`, `.source-modal-backdrop`, `.source-modal-panel`, `.source-modal-header`, `.source-modal-title`, `.source-modal-sub`, `.source-modal-close`, `.source-modal-body`, `.source-modal-loading`. **Keep** `.source-text-pane`, `.source-text-pane mark`, `.source-pdf-toolbar`, `.source-pdf-toolbar button`, `.source-pdf-canvas-wrap`, `.source-pdf-canvas-wrap canvas`, `.source-pdf-highlight-layer`, `.source-pdf-highlight` — these render the actual PDF/TXT content and are reused by the docked viewer in Step 3. Rename the comment above them from `/* ---------- source viewer modal ---------- */` to `/* ---------- docked source viewer ---------- */`.

- [ ] **Step 2: Add the workspace grid**

Add near the top of the file, after the `main{...}` and `.topbar{...}` rules (which stay unchanged):

```css
.workspace-columns{
  display:grid;
  grid-template-columns:1fr 2fr 1fr;
  gap:16px;
  align-items:start;
}
.col{display:flex;flex-direction:column;gap:16px;min-width:0;}

@media (max-width:1100px){
  .workspace-columns{grid-template-columns:1fr;}
}
```

- [ ] **Step 3: Add the docked viewer, chat card, and chat bubble rules**

```css
/* ---------- docked viewer card ---------- */
.viewer-card{flex:1;display:flex;flex-direction:column;min-height:360px;}
.viewer-body{flex:1;overflow:auto;}
.viewer-empty{font-size:12.5px;color:var(--ink-faint);padding:32px 8px;text-align:center;}

/* ---------- chat card ---------- */
.chat-card{display:flex;flex-direction:column;flex:1;min-height:0;}
.ask-freeform{display:flex;gap:8px;margin-top:12px;}
.ask-freeform input{
  flex:1;
  min-width:0;
  border:1px solid var(--line);
  border-radius:100px;
  padding:9px 16px;
  font-size:13px;
  background:var(--card-2);
  color:var(--ink);
}
.ask-freeform input:disabled{color:var(--ink-faint);}
.chat-thread{
  margin-top:16px;
  padding-top:16px;
  border-top:1px solid var(--line);
  display:flex;
  flex-direction:column;
  gap:14px;
  max-height:70vh;
  overflow-y:auto;
}
.chat-empty{font-size:12.5px;color:var(--ink-faint);text-align:center;padding:24px 0;}
.chat-turn{display:flex;flex-direction:column;gap:8px;}
.chat-bubble{border-radius:14px;padding:12px 16px;font-size:13.5px;line-height:1.6;max-width:92%;}
.chat-user{align-self:flex-end;background:var(--dark-card);color:var(--dark-card-ink);}
.chat-assistant{align-self:flex-start;background:var(--card-2);color:var(--ink);width:100%;max-width:100%;box-sizing:border-box;}
.chat-assistant.chat-error{background:var(--brick-soft);color:var(--brick);}
.chat-answer-text{white-space:pre-wrap;}
.chat-thinking-text{display:flex;align-items:center;gap:8px;color:var(--ink-soft);font-size:13px;}
.chat-thinking-text .dots span{animation:chat-dot 1.2s infinite;opacity:0;}
.chat-thinking-text .dots span:nth-child(2){animation-delay:.2s;}
.chat-thinking-text .dots span:nth-child(3){animation-delay:.4s;}
@keyframes chat-dot{0%,80%,100%{opacity:0;}40%{opacity:1;}}
@media (prefers-reduced-motion: reduce){
  .chat-thinking-text .dots span{animation:none;opacity:1;}
}
```

- [ ] **Step 4: Add the Performance panel and Pipeline Parameters disclosure rules**

```css
/* ---------- performance panel ---------- */
.stats-compact{grid-template-columns:repeat(2,1fr);}
.perf-models{display:flex;flex-direction:column;gap:10px;}
.perf-model-row{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:10px 12px;
  border-radius:10px;
  background:var(--card-2);
}
.perf-model-name{font-size:12.5px;font-weight:700;}
.perf-model-tokens{font-size:12.5px;color:var(--ink-soft);}
.perf-empty{font-size:12px;color:var(--ink-faint);}

.submission-card{display:flex;flex-direction:column;}

.params-disclosure summary{
  cursor:pointer;
  font-size:15px;
  font-weight:800;
  letter-spacing:-.01em;
  list-style:none;
}
.params-disclosure summary::-webkit-details-marker{display:none;}
.params-disclosure summary .sub{margin-left:8px;font-weight:400;}
```

- [ ] **Step 5: Bump the stylesheet cache-buster and verify visually**

Confirm `web/index.html`'s `<link rel="stylesheet" href="styles.css?v=workspace-1">` (Task 2 already set this). Hard-refresh `http://127.0.0.1:8080` — expect a styled 3-column layout: left column with upload card + empty viewer card, center column with the chat card (picker, freeform input, empty chat-thread message), right column with compact stats, empty performance panel, submission card, and a collapsed Pipeline Parameters disclosure. Below ~1100px width, columns should stack vertically.

- [ ] **Step 6: Commit**

```bash
git add web/styles.css
git commit -m "feat(web): style the 3-column workspace, chat bubbles, docked viewer; delete dead component CSS"
```

---

### Task 4: JS — dock the source viewer, drop tab/donut/history wiring

**Files:**
- Modify: `web/app.js`

- [ ] **Step 1: Remove the state fields the removed components owned**

In the `state` object, remove `stageTally` and `logFilter` (donut/log-filter only). Remove `expandedHistoryId` (history-table only). Keep `answersById` — still used for evidence rendering when re-rendering a completed turn. Add `modelUsageTotals: new Map()`.

Resulting `state` object:

```js
const state = {
  documentId: null,
  documentFilename: "",
  chunkCount: 0,
  questions: [], // [{question_id, question, category}]
  selectedQuestionId: null,
  submission: null,
  answersById: new Map(), // question_id -> full answer payload (evidence, coverage, ...)
  modelUsageTotals: new Map(), // model -> {input_tokens, output_tokens}
};
```

- [ ] **Step 2: Delete section-tab functions**

Delete `updateSectionTabIndicator`, `setActiveSectionTab`, `initSectionTabs` in full (the `// ---------- wiring ----------` comment stays; these three functions sit right under it).

- [ ] **Step 3: Delete donut and progress-log functions**

Delete `STAGE_INFO`'s use sites are still needed (keep the `STAGE_INFO` constant itself — Task 5 reuses it). Delete: `resetLog`, `appendLogRow`, `applyLogFilter`, `updateProgressStats`, `updateDonut` in full.

- [ ] **Step 4: Delete history-table functions**

Delete `upsertHistoryRow`, `toggleHistoryDetail`, `renderAnswerCard` in full (Task 5 replaces their combined job with per-turn chat rendering). `renderResult` also gets rewritten in Task 5 — leave it alone for now if this task is landing separately, or delete it here if Tasks 4 and 5 land together in one sitting (recommended, since `renderResult` currently calls the functions you just deleted and won't compile-correctly, i.e. won't run without a ReferenceError, until Task 5 replaces it).

- [ ] **Step 5: Refactor `openSourceViewer` to the docked panel — no more modal open/close**

Delete `closeSourceViewer` entirely.

Replace `openSourceViewer` with:

```js
async function openSourceViewer(quote, page) {
  if (!state.documentId) return;
  const empty = el("viewer-empty");
  el("source-pdf-wrap").style.display = "none";
  el("source-text-pane").style.display = "none";
  el("viewer-sub").textContent = page ? `Page ${page}` : "Searching…";

  sourceViewer.targetQuote = coreQuoteText(quote);

  if (sourceViewer.documentId !== state.documentId || !sourceViewer.kind) {
    empty.style.display = "block";
    empty.textContent = "Loading source…";
    try {
      const res = await fetch(`${API_BASE}/documents/${state.documentId}/file`);
      if (!res.ok) throw new Error(`Could not load the source file (${res.status})`);
      const contentType = res.headers.get("content-type") || "";
      sourceViewer.documentId = state.documentId;
      sourceViewer.filename = state.documentFilename;
      if (contentType.includes("pdf")) {
        sourceViewer.kind = "pdf";
        const buffer = await res.arrayBuffer();
        sourceViewer.pdfDoc = await pdfjsLib.getDocument({ data: buffer }).promise;
        sourceViewer.pdfPageCount = sourceViewer.pdfDoc.numPages;
      } else {
        sourceViewer.kind = "txt";
        sourceViewer.txtContent = await res.text();
      }
    } catch (err) {
      empty.textContent = err.message;
      return;
    }
  }

  empty.style.display = "none";
  if (sourceViewer.kind === "pdf") {
    sourceViewer.pdfPage = page && page >= 1 && page <= sourceViewer.pdfPageCount ? page : 1;
    el("source-pdf-wrap").style.display = "block";
    await renderPdfPage();
  } else {
    el("source-text-pane").style.display = "block";
    renderTxtHighlight();
  }
}
```

`renderTxtHighlight` and `renderPdfPage` are unchanged — they already only touch `#source-text-pane` / `#source-pdf-canvas` / `#source-pdf-highlight-layer`, none of which moved.

- [ ] **Step 6: Simplify `initSourceViewer` to just the page-nav buttons**

Replace `initSourceViewer` with:

```js
function initSourceViewer() {
  el("source-pdf-prev").addEventListener("click", () => {
    if (sourceViewer.pdfPage > 1) {
      sourceViewer.pdfPage -= 1;
      renderPdfPage();
    }
  });
  el("source-pdf-next").addEventListener("click", () => {
    if (sourceViewer.pdfPage < sourceViewer.pdfPageCount) {
      sourceViewer.pdfPage += 1;
      renderPdfPage();
    }
  });
}
```

- [ ] **Step 7: Reset the docked viewer when the document is removed**

In `removeDocument`, add before `maybeEnableRunButtons();`:

```js
  sourceViewer.documentId = null;
  sourceViewer.kind = null;
  el("source-pdf-wrap").style.display = "none";
  el("source-text-pane").style.display = "none";
  el("viewer-sub").textContent = "";
  const viewerEmpty = el("viewer-empty");
  viewerEmpty.style.display = "block";
  viewerEmpty.textContent = "Upload a document to preview it here.";
```

- [ ] **Step 8: Remove the pillnav wiring from `initHandlers`**

Delete this block from `initHandlers` (the log-filter pill buttons no longer exist in the DOM):

```js
  document.querySelectorAll(".pillnav button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pillnav button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      state.logFilter = btn.dataset.filter;
      applyLogFilter();
    });
  });
```

- [ ] **Step 9: Update `DOMContentLoaded` to drop `initSectionTabs`**

```js
document.addEventListener("DOMContentLoaded", () => {
  initStatusWidget();
  initSourceViewer();
  initHandlers();
  pollUsage();
  setInterval(pollUsage, 5000);
  loadPipelineSettings();
});
```

- [ ] **Step 10: Confirm it still parses**

Run: `node --check web/app.js`
Expected: no output (syntax OK). It's fine if the page itself is broken in the browser right now — `startRun`/`renderResult` still reference deleted functions until Task 5 lands. Don't load the page in a browser yet if you're doing Tasks 4 and 5 in separate sittings; if landing together, skip straight to Task 5 before testing in a browser.

- [ ] **Step 11: Commit**

```bash
git add web/app.js
git commit -m "refactor(web): dock the source viewer, remove tab/donut/history-table wiring"
```

---

### Task 5: JS — chat-thread rendering and the free-text question input

**Files:**
- Modify: `web/app.js`

- [ ] **Step 1: Add an id generator for freely-typed questions**

Add near the top of the file, after `function el(id) {...}`:

```js
function generateQuestionId() {
  return `q-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}
```

(Picker-selected questions already carry a stable `question_id` from the uploaded file; typed questions need one minted client-side so the chat bubble's DOM id matches the eventual answer's `question_id` in the API response.)

- [ ] **Step 2: Add chat-thread rendering functions**

Add a new section (place it where `upsertHistoryRow`/`renderAnswerCard` used to be):

```js
// ---------- chat thread ----------

function appendChatTurn(questionText, questionId) {
  const emptyState = el("chat-empty");
  if (emptyState) emptyState.remove();

  const thread = el("chat-thread");
  const turn = document.createElement("div");
  turn.className = "chat-turn";
  turn.dataset.qid = questionId;
  turn.innerHTML = `
    <div class="chat-bubble chat-user">${escapeHtml(questionText)}</div>
    <div class="chat-bubble chat-assistant chat-thinking" id="turn-${questionId}">
      <div class="chat-thinking-text">
        <span>Thinking</span>
        <span class="dots"><span>.</span><span>.</span><span>.</span></span>
      </div>
    </div>
  `;
  thread.appendChild(turn);
  thread.scrollTop = thread.scrollHeight;
}

function setThinkingText(questionId, html) {
  const bubble = el(`turn-${questionId}`);
  if (!bubble || !bubble.classList.contains("chat-thinking")) return;
  bubble.innerHTML = `<div class="chat-thinking-text">${html}</div>`;
}

function stageStatusHtml(data) {
  const info = STAGE_INFO[data.stage] || { label: data.stage, badge: null };
  const badgeHtml = info.badge
    ? `<span class="chip ${info.badge === "3B" ? "chip-3b" : "chip-24b"}">${info.badge}</span>`
    : "";
  const failedLabel = data.failed ? `, ${data.failed} failed` : "";
  return `<span>${info.label} (${data.processed}/${data.total}${failedLabel})</span>${badgeHtml}`;
}

function finalizeChatTurn(questionId, answer, question) {
  const bubble = el(`turn-${questionId}`);
  if (!bubble) return;
  bubble.classList.remove("chat-thinking");
  const failed = Boolean(answer.warnings && answer.warnings.length > 0);
  if (failed) bubble.classList.add("chat-error");
  bubble.innerHTML = `
    <div class="chat-answer-text">${escapeHtml(answer.final_answer)}</div>
    <div class="answer-meta">
      <span class="chip chip-cat">${question && question.category ? escapeHtml(question.category) : "uncategorized"}</span>
      ${failed ? `<span class="chip chip-err">${escapeHtml(answer.warnings[0])}</span>` : `<span class="chip chip-ok">done</span>`}
    </div>
    <div class="evidence-list" id="evidence-list-${questionId}"></div>
  `;
  renderEvidenceList(el(`evidence-list-${questionId}`), answer);
}

function errorChatTurn(questionId, message) {
  const bubble = el(`turn-${questionId}`);
  if (!bubble) return;
  bubble.classList.remove("chat-thinking");
  bubble.classList.add("chat-error");
  bubble.innerHTML = `<div class="chat-answer-text">${escapeHtml(message)}</div>`;
}
```

Note: `renderEvidenceList`'s call to `openSourceViewer` (inside the existing `renderEvidenceList` function you're not modifying) still works unchanged — it already calls `openSourceViewer(item.quote, ...)`, which Task 4 already repointed at the docked panel.

- [ ] **Step 3: Rewrite `startRun`, `streamProgress`, and `renderResult`**

Replace all three (they currently target the single global log/answer-card/history-table — the new versions target per-turn chat bubbles):

```js
async function startRun(questions) {
  clearError();
  questions.forEach((q) => appendChatTurn(q.question, q.question_id));
  if (questions.length > 1) {
    // The SSE stream reports one aggregate {stage, processed, total} for the
    // whole batch, not per-question -- show that live status on the most
    // recent turn only, and mark the rest as queued so nothing looks stuck.
    questions.slice(0, -1).forEach((q) => setThinkingText(q.question_id, "<span>Queued…</span>"));
  }
  try {
    const res = await fetch(`${API_BASE}/answer-jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: state.documentId, questions }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Failed to start run (${res.status})`);
    }
    const job = await res.json();
    streamProgress(job.job_id, questions);
  } catch (err) {
    questions.forEach((q) => errorChatTurn(q.question_id, err.message));
  }
}

function streamProgress(jobId, questions) {
  const activeId = questions[questions.length - 1].question_id;
  const source = new EventSource(`${API_BASE}/answer-jobs/${jobId}/stream`);

  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (!data.stage) return; // final-state replay payload has no "stage" guarantee; skip
    setThinkingText(activeId, stageStatusHtml(data));
    el("stat-progress").textContent = `${data.processed}/${data.total}`;
  };

  source.addEventListener("done", async () => {
    source.close();
    try {
      const res = await fetch(`${API_BASE}/answer-jobs/${jobId}`);
      const job = await res.json();
      if (job.status === "completed") {
        renderResult(job.result, questions);
      } else {
        questions.forEach((q) => errorChatTurn(q.question_id, job.error || "The run failed."));
      }
    } catch (err) {
      questions.forEach((q) => errorChatTurn(q.question_id, `Failed to load results: ${err.message}`));
    }
  });

  source.onerror = () => {
    questions.forEach((q) => {
      const bubble = el(`turn-${q.question_id}`);
      if (bubble && bubble.classList.contains("chat-thinking")) {
        setThinkingText(q.question_id, "<span>Connection lost — reload to check the job's status; it's still running server-side.</span>");
      }
    });
  };
}

function renderResult(result, questions) {
  mergeSubmission(result.submission);
  el("download-submission-btn").disabled = false;
  updateSubmissionProgressLabel();

  if (result.usage) {
    const total = (result.usage.total_input_tokens || 0) + (result.usage.total_output_tokens || 0);
    el("stat-tokens").textContent = total.toLocaleString("en-US");
    accumulateModelUsage(result.usage.by_model || {});
  }

  const byId = new Map(questions.map((q) => [q.question_id, q]));
  result.answers.forEach((answer) => {
    const question = byId.get(answer.question_id) || state.questions.find((q) => q.question_id === answer.question_id);
    state.answersById.set(answer.question_id, answer);
    finalizeChatTurn(answer.question_id, answer, question);
  });
}
```

- [ ] **Step 4: Add the free-text submit handler and Performance-panel accumulation**

Add near `getSelectedQuestion`:

```js
function submitFreeformQuestion() {
  const input = el("freeform-input");
  const text = input.value.trim();
  if (!text) return;
  const question = { question_id: generateQuestionId(), question: text, category: "" };
  input.value = "";
  startRun([question]);
}
```

Add near `updateDonut`'s old spot (now deleted — put this in the same general area, after `formatUsageCost`/`pollUsage`, before `// ---------- pipeline parameters`):

```js
// ---------- performance panel (real per-model token totals) ----------

function accumulateModelUsage(byModel) {
  Object.entries(byModel).forEach(([model, usage]) => {
    const current = state.modelUsageTotals.get(model) || { input_tokens: 0, output_tokens: 0 };
    current.input_tokens += usage.input_tokens || 0;
    current.output_tokens += usage.output_tokens || 0;
    state.modelUsageTotals.set(model, current);
  });
  renderPerformanceStats();
}

function renderPerformanceStats() {
  const container = el("perf-models");
  if (state.modelUsageTotals.size === 0) {
    container.innerHTML = '<div class="perf-empty">Run a question to see the model breakdown.</div>';
    return;
  }
  const rows = Array.from(state.modelUsageTotals.entries()).map(([model, usage]) => {
    const total = usage.input_tokens + usage.output_tokens;
    return `
      <div class="perf-model-row">
        <span class="perf-model-name mono">${escapeHtml(model)}</span>
        <span class="perf-model-tokens mono">${total.toLocaleString("en-US")} tok</span>
      </div>
    `;
  });
  container.innerHTML = rows.join("");
}
```

- [ ] **Step 5: Wire the free-text input and enable/disable it correctly**

In `maybeEnableRunButtons`, add the freeform button:

```js
function maybeEnableRunButtons() {
  const hasDocument = Boolean(state.documentId);
  const hasQuestions = state.questions.length > 0;
  const hasSelection = Boolean(getSelectedQuestion());
  el("run-selected-btn").disabled = !(hasDocument && hasSelection);
  el("run-all-btn").disabled = !(hasDocument && hasQuestions);
  el("freeform-ask-btn").disabled = !hasDocument;
  el("freeform-input").disabled = !hasDocument;
}
```

In `initHandlers`, add (anywhere alongside the other `el(...).addEventListener` calls):

```js
  el("freeform-ask-btn").addEventListener("click", submitFreeformQuestion);
  el("freeform-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !el("freeform-ask-btn").disabled) submitFreeformQuestion();
  });
```

- [ ] **Step 6: Confirm it parses**

Run: `node --check web/app.js`
Expected: no output.

- [ ] **Step 7: Manual verification in the browser**

Start both servers (see "Before you start"). Hard-refresh `http://127.0.0.1:8080`, then walk through the four states called out in the spec's Testing section:

1. **Empty state** — no document uploaded: freeform input and both run buttons disabled, viewer card shows "Upload a document to preview it here.", chat thread shows the empty-state message, Performance panel shows "Run a question to see the model breakdown."
2. **Thinking state** — upload a document, type a question into the freeform input, hit Enter or click Ask: a user bubble appears immediately, followed by an assistant bubble showing "Thinking..." then live stage text with a model badge as the SSE events arrive.
3. **Completed turn with multi-page evidence** — once answered, click "View in source" on two different evidence chips that have different page numbers; confirm the left panel's viewer jumps to the correct page and highlights it each time (no modal opens — the left panel updates in place).
4. **Failed turn** — trigger a failure (e.g. stop the backend mid-run, or submit while the LLM broker is down) and confirm that turn's assistant bubble renders in the red error state in place, without breaking the rest of the thread.

Also confirm: uploading a question set still populates the picker and "Run this question"/"Run all" still work and append turns the same way; removing the document (×) resets the viewer to its empty state; the Pipeline Parameters `<details>` expands to show the 9 real fields; the Performance panel accumulates token counts by model across more than one run.

- [ ] **Step 8: Commit**

```bash
git add web/app.js
git commit -m "feat(web): chat-thread rendering, free-text question input, performance panel"
```

---

### Task 6: Final pass

**Files:** none (verification only)

- [ ] **Step 1: Full backend test suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: all passing.

- [ ] **Step 2: Re-read the spec's Non-goals and confirm none were accidentally implemented**

Check: no resizable column dividers were added, no cache-hit-rate stat was added, no chat-history persistence (localStorage etc.) was added, no multi-document support was added, `evidence_quotes`/`source_pages` in the API response are untouched.

- [ ] **Step 3: Grep for leftover references to deleted DOM ids**

Run: `grep -n "log-list\|section-tab\|history-body\|source-modal\|donut\|pillnav" web/app.js web/index.html`
Expected: no matches (if anything matches, it's dead code or a missed rename — fix it before calling this done).

- [ ] **Step 4: Final commit if Step 2/3 required fixes**

```bash
git add -A
git commit -m "fix(web): clean up remaining references from the workspace redesign"
```

(Skip this commit if Steps 2-3 found nothing.)
