# Connecting the dashboard to a different pipeline branch

This `web/` folder (`index.html`, `styles.css`, `app.js`) is a static,
framework-free frontend. It never imports pipeline code directly -- it only
calls HTTP endpoints on a FastAPI backend. That means it can sit in front of
**any** teammate's pipeline branch, as long as that branch's
`app/api/main.py` exposes the same endpoints in the same shape.

## Steps for a teammate's machine

1. Copy this `web/` folder into their project root (next to their own `app/`).
2. Make sure their backend exposes the endpoints listed below.
3. Run their backend: `uvicorn app.api.main:app --host 127.0.0.1 --port <PORT>`.
4. Run the frontend from inside `web/`: `python -m http.server <STATIC_PORT>`.
5. Open the dashboard in a browser, and in the "Model backend" dropdown
   (top of the right column in `index.html`) either pick an existing
   option or add a new `<option value="http://127.0.0.1:<PORT>">Name</option>`.

## Required endpoints (must exist, same request/response shape)

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /health/llm` | status checks |
| `GET /usage` | broker cost/usage, polled every 5s |
| `GET /pipeline-settings` | read-only config shown in the UI |
| `POST /documents` | upload + parse, returns `{document_id, filename, page_count, chunk_count}` |
| `GET /documents/{id}` | metadata |
| `GET /documents/{id}/file` | **new** -- streams the original upload back (for the docked PDF/TXT viewer). See "Additive patches" below. |
| `POST /answer-jobs` | starts a background job, returns `{job_id}` |
| `GET /answer-jobs/{id}` | job status / final result |
| `GET /answer-jobs/{id}/stream` | SSE progress: `data: {stage, processed, total, failed}` per event, then `event: done` |
| `GET /runs/{id}/submission` | grader-format JSON |

## Additive patches this branch made on top of the base pipeline

None of these change answer content -- they only add fields/endpoints the
UI reads. If a teammate's pipeline doesn't have them yet, port these three
small changes (see this branch's own `app/` for a working reference):

1. **`app/schemas.py`** -- add an `EvidenceQuote {quote: str, page: int|None}`
   model, and `evidence_quotes: list[EvidenceQuote]` +
   `source_pages: list[int]` fields on `PipelineAnswer`.
2. **`app/pipeline.py`** -- in whatever function builds the final
   `PipelineAnswer` per question, pair each evidence string with a page
   (falling back to a regex match on an embedded `"(page N)"` substring if
   the pipeline doesn't already track per-quote pages) and populate the two
   fields above.
3. **`app/api/main.py`**:
   - include `evidence_quotes` / `source_pages` when serializing each
     answer for `/answer-jobs` and `/answer`.
   - add `by_model: {model: {input_tokens, output_tokens}}` to the `usage`
     block (sum whatever per-call usage records the pipeline already
     tracks, grouped by model).
   - add the `GET /documents/{id}/file` route: look up the document's
     stored filename, serve it back with `FileResponse` (content-type
     `application/pdf` or `text/plain` based on the extension).

Without these three, the dashboard still works end-to-end (upload, ask,
chat thread, performance stats) -- it just won't show evidence quotes/page
highlights or the per-model token breakdown for that backend.

## Known limitation

Answering against a PDF is measurably less accurate than answering against
the same document's `.txt` export (a confirmed drop from ~69% to ~30%s on
the practice set), and page numbers are not guaranteed to line up between a
`.txt` and a `.pdf` of the same book -- so cross-format viewing (answer
from `.txt`, display a separately-uploaded `.pdf`) can jump to the wrong
page. Prefer `.txt` for both answering and display when accuracy or
highlight correctness matters.
