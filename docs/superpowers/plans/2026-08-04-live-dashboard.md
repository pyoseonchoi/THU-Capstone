# Live Dashboard Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a static, no-build-step dashboard (`web/`) on top of the existing FULLSCAN-QA FastAPI backend, giving live document/question-set upload, one-click question execution, a real-time SSE progress log with model-routing badges, a cost counter, and a `submission.json` download — with zero changes to pipeline/mapping logic.

**Architecture:** `app/api/main.py` gets four small additive changes (CORS middleware, a per-job `asyncio.Queue`, a queue-push inside the existing progress closure, one new SSE route). Everything else lives in a brand-new `web/` folder (`index.html` + `styles.css` + `app.js`, no framework, no bundler) that talks to the backend over `fetch`/`EventSource` from a separate static server.

**Tech Stack:** FastAPI/Starlette (existing), `asyncio.Queue` + `StreamingResponse` for SSE, plain HTML/CSS/JS on the frontend, pytest + `pytest-asyncio` (already configured, `asyncio_mode = "auto"`) for backend tests.

**Spec:** `docs/superpowers/specs/2026-08-04-live-dashboard-design.md` (approved, 2 review passes).

---

## File Structure

```
app/api/main.py          MODIFY — CORS, job queue, SSE route (only file touched in app/)
tests/test_api.py        MODIFY — tests for the above
web/index.html            CREATE — dashboard markup
web/styles.css            CREATE — dashboard styling (ported from the approved wireframe)
web/app.js                 CREATE — upload, run, SSE, rendering, usage polling
```

No other files in `app/`, `ui/`, or `scripts/` are touched. `web/` is served independently
(`python -m http.server 5500` from inside `web/`) against the backend on
`http://127.0.0.1:8000` (started separately with `uvicorn app.api.main:app --port 8000`).

---

## Task 1: CORS middleware

**Files:**
- Modify: `app/api/main.py:1-30`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
class TestCORS:
    def test_cors_allows_cross_origin_requests(self, client):
        response = client.get("/health", headers={"Origin": "http://127.0.0.1:5500"})
        assert response.headers.get("access-control-allow-origin") == "*"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py::TestCORS -v`
Expected: FAIL — no `access-control-allow-origin` header present.

- [ ] **Step 3: Add the middleware**

In `app/api/main.py`, add the import next to the other `fastapi` imports (line 8):

```python
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
```

Immediately after the `app = FastAPI(...)` block (after line 30), add:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

(`allow_origins=["*"]` is fine here — this is a local dev/demo server behind the course
broker, not a public deployment. `StreamingResponse` is imported now because Task 3 needs it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py::TestCORS -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/main.py tests/test_api.py
git commit -m "feat(api): add CORS middleware for the local dashboard frontend"
```

---

## Task 2: Per-job progress queue

**Files:**
- Modify: `app/api/main.py:196-257`
- Test: `tests/test_api.py`

> Line numbers below (and in Task 3) are computed against `app/api/main.py` as it stood
> *before* Task 1's edits. Task 1 adds ~9 lines above this section, so the real line numbers
> will have shifted by the time you get here. Locate each edit by matching the quoted code
> snippet's content (e.g. search for `_answer_jobs: dict[str, dict] = {}`), not by trusting
> the line number literally.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py` (needs `import asyncio` and `from app.schemas import PipelineAnswer,
PipelineRun, QuestionRequest` added to the file's imports):

```python
import asyncio

from app.schemas import PipelineAnswer, PipelineRun, QuestionRequest


class TestAnswerJobProgressQueue:
    async def test_progress_events_are_queued_in_order(self, monkeypatch):
        class FakeJobPipeline:
            async def answer_questions(self, document_id, questions, mode=None, progress_callback=None):
                if progress_callback:
                    progress_callback("mapping", 1, 2, 0)
                    progress_callback("answering", 2, 2, 0)
                return PipelineRun(
                    document_id=document_id,
                    answers=[PipelineAnswer(question_id="q1", final_answer="answer")],
                )

            async def close(self):
                return None

        async def fake_llm_health(settings):
            return {"available": True, "message": "ok"}

        monkeypatch.setattr(api_main, "_get_pipeline", lambda: FakeJobPipeline())
        monkeypatch.setattr(api_main, "check_llm_health", fake_llm_health)

        job_id = "job-progress-1"
        api_main._answer_jobs[job_id] = {
            "job_id": job_id, "status": "running", "stage": "compiling",
            "processed": 0, "total": 0, "failed": 0,
        }
        api_main._answer_job_queues[job_id] = asyncio.Queue()

        req = api_main.AnswerRequest(
            document_id="doc-1",
            questions=[QuestionRequest(question_id="q1", question="test?")],
        )
        await api_main._execute_answer_job(job_id, req)

        queue = api_main._answer_job_queues[job_id]
        events = []
        while True:
            item = queue.get_nowait()
            if item is None:
                break
            events.append(item)

        assert events == [
            {"stage": "mapping", "processed": 1, "total": 2, "failed": 0},
            {"stage": "answering", "processed": 2, "total": 2, "failed": 0},
        ]
        assert api_main._answer_jobs[job_id]["status"] == "completed"
```

(`asyncio_mode = "auto"` in `pyproject.toml` means this `async def test_...` runs directly,
no `@pytest.mark.asyncio` needed.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py::TestAnswerJobProgressQueue -v`
Expected: FAIL — `AttributeError: module 'app.api.main' has no attribute '_answer_job_queues'`

- [ ] **Step 3: Add the queue and wire it into the progress closure**

In `app/api/main.py`, change line 196 from:

```python
_answer_jobs: dict[str, dict] = {}
```

to:

```python
_answer_jobs: dict[str, dict] = {}
_answer_job_queues: dict[str, asyncio.Queue] = {}
```

Replace the `progress` closure (lines 208-214):

```python
        def progress(stage: str, processed: int, total: int, failed: int) -> None:
            _answer_jobs[job_id].update({
                "stage": stage,
                "processed": processed,
                "total": total,
                "failed": failed,
            })
```

with:

```python
        queue = _answer_job_queues.get(job_id)

        def progress(stage: str, processed: int, total: int, failed: int) -> None:
            event = {
                "stage": stage,
                "processed": processed,
                "total": total,
                "failed": failed,
            }
            _answer_jobs[job_id].update(event)
            if queue is not None:
                queue.put_nowait(event)
```

Replace the `finally` block (lines 234-236):

```python
    finally:
        if pipeline is not None:
            await pipeline.close()
```

with:

```python
    finally:
        if pipeline is not None:
            await pipeline.close()
        if queue is not None:
            queue.put_nowait(None)
```

Note: `queue = _answer_job_queues.get(job_id)` must be defined before the `try` block's
`def progress(...)` uses it, and the `finally` block also needs it in scope — put the
`queue = _answer_job_queues.get(job_id)` line right after `pipeline: FullScanPipeline | None = None`
(before the `try:`), not inside it.

In `create_answer_job`, right after `_answer_jobs[job_id] = {...}` (after line 255), add:

```python
    _answer_job_queues[job_id] = asyncio.Queue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py::TestAnswerJobProgressQueue -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest tests/ -v -m "not live"`
Expected: all PASS (existing `TestAnswerJobs`/`TestDocumentUpload`/`TestHealthEndpoint` unaffected)

- [ ] **Step 6: Commit**

```bash
git add app/api/main.py tests/test_api.py
git commit -m "feat(api): queue progress events per answer job"
```

---

## Task 3: SSE stream route

**Files:**
- Modify: `app/api/main.py` (add route after `get_answer_job`, currently ending at line 266)
- Test: `tests/test_api.py`

> Same caveat as Task 2: this line number is pre-Task-1/2. Find `get_answer_job`'s closing
> `return job` by content and add the new route right after it, not at a literal line 266.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`:

```python
from fastapi import HTTPException


class TestAnswerJobStream:
    async def test_stream_emits_queued_events_then_done(self):
        job_id = "job-stream-1"
        api_main._answer_jobs[job_id] = {"job_id": job_id, "status": "running"}
        queue = asyncio.Queue()
        queue.put_nowait({"stage": "mapping", "processed": 1, "total": 2, "failed": 0})
        queue.put_nowait(None)
        api_main._answer_job_queues[job_id] = queue

        response = await api_main.stream_answer_job(job_id)
        chunks = [chunk async for chunk in response.body_iterator]
        body = "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

        assert '"stage": "mapping"' in body
        assert body.strip().endswith("event: done\ndata: {}")

    async def test_stream_of_unknown_job_returns_404(self):
        with pytest.raises(HTTPException) as exc_info:
            await api_main.stream_answer_job("does-not-exist")
        assert exc_info.value.status_code == 404

    async def test_stream_of_finished_job_replays_final_state_immediately(self):
        job_id = "job-stream-2"
        api_main._answer_jobs[job_id] = {
            "job_id": job_id, "status": "completed", "stage": "answering",
        }
        # no queue registered -- simulates a reconnect after the job already finished
        api_main._answer_job_queues.pop(job_id, None)

        response = await api_main.stream_answer_job(job_id)
        chunks = [chunk async for chunk in response.body_iterator]
        body = "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)

        assert '"status": "completed"' in body
        assert "event: done" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py::TestAnswerJobStream -v`
Expected: FAIL — `AttributeError: module 'app.api.main' has no attribute 'stream_answer_job'`

- [ ] **Step 3: Add the route**

In `app/api/main.py`, add this immediately after `get_answer_job` (after line 266):

```python
@app.get("/answer-jobs/{job_id}/stream")
async def stream_answer_job(job_id: str):
    """Stream progress events for an answer job as Server-Sent Events."""
    job = _answer_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Answer job not found")
    queue = _answer_job_queues.get(job_id)

    async def event_source():
        import json

        if queue is None:
            yield f"data: {json.dumps(job)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        while True:
            event = await queue.get()
            if event is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api.py::TestAnswerJobStream -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v -m "not live"`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add app/api/main.py tests/test_api.py
git commit -m "feat(api): add SSE route for answer-job progress"
```

This is the last backend task — `app/api/main.py` now has everything `web/` needs.

---

## Task 4: Dashboard HTML + CSS shell

**Files:**
- Create: `web/index.html`
- Create: `web/styles.css`

No automated tests for markup (per spec: no frontend test framework). Verification is
visual, in Step 3.

- [ ] **Step 1: Create `web/styles.css`**

```css
:root{
  --bg:#eeeef0;
  --card:#ffffff;
  --card-2:#f5f5f7;
  --ink:#17181c;
  --ink-soft:#8b8e95;
  --ink-faint:#b3b6bc;
  --line:#e8e8eb;
  --accent:#1f6f63;
  --accent-ink:#12463d;
  --accent-soft:#e2f0ec;
  --dark-card:#15171a;
  --dark-card-ink:#f4f4f2;
  --dark-card-soft:#8d9199;
  --good:#1f9d55;
  --bad:#d1453b;
  --pill-bg:#eceef0;
  --pill-ink:#3a3d42;
  --amber:#a9781f;
  --amber-soft:#f8efdb;
  --slate:#3f6fa8;
  --slate-soft:#e8eff7;
  --brick:#c14c3f;
  --brick-soft:#fbe7e4;
  --shadow:0 1px 2px rgba(20,20,25,.04), 0 10px 24px -8px rgba(20,20,25,.10);
}
:root[data-theme="dark"]{
  --bg:#0f1113; --card:#1a1d21; --card-2:#22262b; --ink:#edeef0; --ink-soft:#9a9fa7;
  --ink-faint:#5d6169; --line:#2a2e33; --accent:#63b5a5; --accent-ink:#bfe9de;
  --accent-soft:#1c2b28; --dark-card:#050607; --dark-card-ink:#f4f4f2; --dark-card-soft:#888d95;
  --good:#4ade80; --bad:#f78b81; --pill-bg:#2a2d31; --pill-ink:#d7dade;
  --amber:#d9a955; --amber-soft:#332a16; --slate:#8fb4dd; --slate-soft:#1e2733;
  --brick:#e08e83; --brick-soft:#3a221e;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 12px 26px -10px rgba(0,0,0,.5);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0f1113; --card:#1a1d21; --card-2:#22262b; --ink:#edeef0; --ink-soft:#9a9fa7;
    --ink-faint:#5d6169; --line:#2a2e33; --accent:#63b5a5; --accent-ink:#bfe9de;
    --accent-soft:#1c2b28; --dark-card:#050607; --dark-card-ink:#f4f4f2; --dark-card-soft:#888d95;
    --good:#4ade80; --bad:#f78b81; --pill-bg:#2a2d31; --pill-ink:#d7dade;
    --amber:#d9a955; --amber-soft:#332a16; --slate:#8fb4dd; --slate-soft:#1e2733;
    --brick:#e08e83; --brick-soft:#3a221e;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 12px 26px -10px rgba(0,0,0,.5);
  }
}

*{box-sizing:border-box;}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Pretendard,Roboto,"Malgun Gothic",sans-serif;
  line-height:1.45;
}
.mono{font-family:ui-monospace,"Cascadia Mono","SF Mono",Consolas,monospace;font-variant-numeric:tabular-nums;}
svg{display:block;}
a{color:var(--accent);}
button,input,select{font:inherit;}
button:focus-visible,input:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}

.app{display:flex;min-height:100vh;}

.sidebar{width:232px;flex:none;padding:24px 16px;display:flex;flex-direction:column;gap:22px;}
.brand{display:flex;align-items:center;gap:10px;padding:0 8px;}
.brand .mark{
  width:32px;height:32px;border-radius:9px;background:var(--dark-card);color:var(--dark-card-ink);
  display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;letter-spacing:-.02em;
}
.brand span{font-weight:800;font-size:16.5px;letter-spacing:-.01em;}

.index-card{margin-top:auto;background:var(--dark-card);color:var(--dark-card-ink);border-radius:14px;padding:16px;}
.index-card .t{font-weight:700;font-size:13px;margin-bottom:4px;}
.index-card .d{font-size:11.5px;color:var(--dark-card-soft);line-height:1.5;margin-bottom:12px;}
.index-card .bar{height:5px;border-radius:100px;background:rgba(255,255,255,.14);overflow:hidden;margin-bottom:8px;}
.index-card .bar i{display:block;height:100%;background:var(--accent);width:0%;border-radius:100px;transition:width .3s ease;}
.index-card .stat{font-size:11px;color:var(--dark-card-soft);}
.index-card .stat b{color:var(--dark-card-ink);font-family:ui-monospace,monospace;}

main{flex:1;min-width:0;padding:26px 32px 56px;}
.topbar{display:flex;align-items:center;gap:18px;margin-bottom:22px;}
.topbar h1{font-size:23px;font-weight:800;letter-spacing:-.01em;margin:0;flex:none;}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px;}
.stat-card{background:var(--card);border-radius:16px;padding:18px 18px;box-shadow:var(--shadow);}
.stat-card.dark{background:var(--dark-card);color:var(--dark-card-ink);}
.stat-card .l{font-size:12px;color:var(--ink-soft);margin-bottom:10px;}
.stat-card.dark .l{color:var(--dark-card-soft);}
.stat-card .n{font-size:23px;font-weight:800;letter-spacing:-.01em;font-family:ui-monospace,monospace;}
.stat-card .delta{display:flex;align-items:center;gap:4px;margin-top:8px;font-size:11.5px;color:var(--ink-soft);}
.stat-card.dark .delta{color:var(--dark-card-soft);}
.delta .n2{color:inherit;font-weight:700;}

.card{background:var(--card);border-radius:16px;padding:20px;box-shadow:var(--shadow);}
.card-head{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
.card-head h2{font-size:15px;font-weight:800;margin:0;letter-spacing:-.01em;}
.card-head .sub{font-size:11.5px;color:var(--ink-faint);}
.pillnav{display:flex;gap:4px;background:var(--card-2);padding:3px;border-radius:100px;margin-left:auto;}
.pillnav button{border:none;background:transparent;border-radius:100px;padding:6px 13px;font-size:12px;font-weight:600;color:var(--ink-soft);cursor:pointer;}
.pillnav button.on{background:var(--dark-card);color:var(--dark-card-ink);}

.row2{display:grid;grid-template-columns:1.15fr .85fr;gap:14px;margin-bottom:14px;}
@media (max-width:980px){.row2{grid-template-columns:1fr;} .stats{grid-template-columns:repeat(2,1fr);}}

.dropzone{border:1.5px dashed var(--line);border-radius:14px;background:var(--card-2);padding:22px;text-align:center;}
.dropzone .icon{width:38px;height:38px;border-radius:50%;background:var(--accent-soft);color:var(--accent-ink);display:flex;align-items:center;justify-content:center;margin:0 auto 10px;}
.dropzone .icon svg{width:18px;height:18px;}
.dropzone .t{font-size:13px;font-weight:700;}
.dropzone .d{font-size:11.5px;color:var(--ink-faint);margin-top:3px;}
.browse{display:inline-block;margin-top:12px;background:var(--dark-card);color:var(--dark-card-ink);border-radius:100px;padding:7px 16px;font-size:12px;font-weight:700;cursor:pointer;}

.filelist{margin-top:14px;display:flex;flex-direction:column;gap:8px;}
.file{display:flex;align-items:center;gap:10px;padding:8px 4px;}
.file .ext{width:34px;height:34px;border-radius:9px;flex:none;display:flex;align-items:center;justify-content:center;font-size:9.5px;font-weight:800;color:#fff;}
.file .ext.txt{background:var(--accent);}
.file .ext.yaml{background:#7a7f88;}
.file .name{font-size:12.5px;font-weight:600;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.file .size{font-size:11px;color:var(--ink-faint);flex:none;}

.chunk-summary{margin-top:14px;padding-top:14px;border-top:1px solid var(--line);display:flex;align-items:center;gap:10px;font-size:12px;color:var(--ink-soft);}
.chunk-summary b{color:var(--ink);font-family:ui-monospace,monospace;}

.row{display:flex;align-items:center;gap:10px;margin-top:11px;flex-wrap:wrap;}
select{background:var(--slate-soft);color:var(--slate);border:none;border-radius:100px;padding:7px 14px;font-size:12.5px;font-weight:700;flex:1;min-width:0;}
select:disabled{opacity:.6;}
.spacer{flex:1;}
button.primary{background:var(--dark-card);color:var(--dark-card-ink);border:none;border-radius:100px;padding:9px 20px;font-size:13px;font-weight:700;cursor:pointer;}
button.primary:disabled{opacity:.4;cursor:not-allowed;}
.answer{margin-top:16px;padding-top:16px;border-top:1px solid var(--line);}
.answer .t{font-size:13.5px;line-height:1.65;}
.answer-meta{display:flex;gap:7px;margin-top:11px;flex-wrap:wrap;}

.chip{display:inline-flex;align-items:center;gap:5px;font-size:10.5px;font-weight:700;border-radius:100px;padding:4px 10px;white-space:nowrap;}
.chip-cat{background:var(--pill-bg);color:var(--pill-ink);}
.chip-3b{background:var(--slate-soft);color:var(--slate);}
.chip-24b{background:var(--amber-soft);color:var(--amber);}
.chip-ok{background:var(--accent-soft);color:var(--accent-ink);}
.chip-err{background:var(--brick-soft);color:var(--brick);}

.log{list-style:none;margin:0;padding:0;max-height:360px;overflow-y:auto;}
.log li{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-bottom:1px solid var(--line);}
.log li:last-child{border-bottom:none;}
.dot{width:8px;height:8px;border-radius:50%;margin-top:6px;flex:none;background:var(--good);}
.dot.pending{background:var(--ink-faint);}
.dot.active{background:var(--amber);animation:pulse 1.4s ease-in-out infinite;}
@media (prefers-reduced-motion: reduce){.dot.active{animation:none;}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(169,120,31,.35);}50%{box-shadow:0 0 0 5px rgba(169,120,31,0);}}
.log .step{font-size:13px;font-weight:500;}
.log .step.pending{color:var(--ink-faint);}
.log .meta{display:flex;align-items:center;gap:8px;margin-top:5px;}
.log .time{font-size:11px;color:var(--ink-faint);}

.donut-wrap{display:flex;align-items:center;gap:18px;}
.donut{width:104px;height:104px;border-radius:50%;flex:none;display:flex;align-items:center;justify-content:center;transition:background .3s ease;}
.donut::after{content:"";width:70px;height:70px;border-radius:50%;background:var(--card);}
.donut-legend{display:flex;flex-direction:column;gap:10px;}
.legend-row{display:flex;align-items:center;gap:8px;font-size:12.5px;}
.swatch{width:9px;height:9px;border-radius:3px;flex:none;}
.legend-row .n{margin-left:auto;font-weight:700;font-family:ui-monospace,monospace;}

.history{margin-top:14px;}
.table-wrap{overflow-x:auto;border-radius:16px;background:var(--card);box-shadow:var(--shadow);}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:560px;}
thead th{text-align:left;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-faint);padding:14px 20px;border-bottom:1px solid var(--line);}
tbody td{padding:13px 20px;border-bottom:1px solid var(--line);vertical-align:middle;}
tbody tr:last-child td{border-bottom:none;}
.qid{color:var(--ink-faint);}
.qtext{max-width:360px;}
.status-pill{background:var(--pill-bg);color:var(--pill-ink);border-radius:100px;padding:4px 12px;font-size:11px;font-weight:700;display:inline-block;}
.status-pill.ok{background:var(--dark-card);color:var(--dark-card-ink);}

.error-banner{margin-top:12px;padding:10px 14px;border-radius:10px;background:var(--brick-soft);color:var(--brick);font-size:12.5px;display:none;}
```

- [ ] **Step 2: Create `web/index.html`**

```html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long-Context Challenge — 파이프라인 대시보드</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<div class="app">

  <aside class="sidebar">
    <div class="brand"><div class="mark">LC</div><span>LongContext</span></div>

    <div class="index-card">
      <div class="t">문서 상태</div>
      <div class="d" id="index-doc-label">문서를 업로드하세요</div>
      <div class="bar"><i id="index-bar"></i></div>
      <div class="stat">청크 <b id="index-chunk-count">0 / 0</b></div>
    </div>
  </aside>

  <main>
    <div class="topbar">
      <h1>파이프라인 대시보드</h1>
    </div>

    <div class="stats">
      <div class="stat-card dark">
        <div class="l">누적 비용</div>
        <div class="n" id="stat-cost">—</div>
        <div class="delta"><span class="n2" id="stat-progress">0/0</span>&nbsp;문항 진행</div>
      </div>
      <div class="stat-card">
        <div class="l">토큰 (입력+출력)</div>
        <div class="n" id="stat-tokens">—</div>
        <div class="delta">최근 실행 기준</div>
      </div>
      <div class="stat-card">
        <div class="l">색인 청크</div>
        <div class="n" id="stat-chunks">0</div>
        <div class="delta" id="stat-chunks-status">업로드 대기</div>
      </div>
      <div class="stat-card">
        <div class="l">질문 세트</div>
        <div class="n" id="stat-question-count">0</div>
        <div class="delta">로드된 문항 수</div>
      </div>
    </div>

    <div class="row2">
      <div class="card">
        <div class="card-head"><h2>문서 &amp; 질문 세트</h2><span class="sub">Upload</span></div>

        <div class="dropzone" id="doc-dropzone">
          <div class="icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 4v11M12 15l7-4M12 15l-7-4M5 11V7l7-4 7 4v4"/></svg>
          </div>
          <div class="t">문서를 끌어다 놓거나 클릭해서 업로드</div>
          <div class="d">.txt · .pdf — 최대 32MB</div>
          <label class="browse" for="doc-input">문서 선택</label>
          <input type="file" id="doc-input" accept=".txt,.pdf" hidden>
        </div>

        <div class="row">
          <label class="browse" for="questions-input">질문 세트(JSON) 선택</label>
          <input type="file" id="questions-input" accept=".json" hidden>
          <span class="sub" id="questions-count-label">아직 업로드 안 됨</span>
        </div>

        <div class="filelist" id="file-list"></div>

        <div class="chunk-summary" id="chunk-summary" style="display:none;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>
          <span><b id="chunk-summary-count">0</b>개 청크로 분할 · 문서 업로드 완료</span>
        </div>

        <div class="error-banner" id="upload-error"></div>
      </div>

      <div class="card">
        <div class="card-head"><h2>질문 실행</h2><span class="sub">Ask</span></div>

        <div class="row" style="margin-top:0;">
          <select id="question-picker" disabled>
            <option>질문 세트를 먼저 업로드하세요</option>
          </select>
        </div>
        <div class="row">
          <span class="chip chip-cat" id="picked-category">미분류</span>
          <span class="spacer"></span>
          <button class="primary" id="run-selected-btn" disabled>이 질문만 실행</button>
          <button class="primary" id="run-all-btn" disabled>전체 실행</button>
        </div>

        <div class="answer" id="answer-block" style="display:none;">
          <div class="t" id="answer-text"></div>
          <div class="answer-meta" id="answer-meta"></div>
        </div>
      </div>
    </div>

    <div class="row2">
      <div class="card">
        <div class="card-head">
          <h2>진행 로그</h2>
          <div class="pillnav">
            <button class="on" data-filter="all">전체</button>
            <button data-filter="3B">3B</button>
            <button data-filter="24B">24B</button>
          </div>
        </div>
        <ul class="log" id="log-list">
          <li><div class="dot pending"></div><div><div class="step pending">문서와 질문 세트를 업로드하고 실행하면 여기에 진행 로그가 표시됩니다</div></div></li>
        </ul>
      </div>

      <div class="card">
        <div class="card-head"><h2>모델 사용 비율</h2><span class="sub">이번 세션</span></div>
        <div class="donut-wrap">
          <div class="donut" id="donut" style="background:conic-gradient(var(--line) 0 100%);"></div>
          <div class="donut-legend">
            <div class="legend-row"><span class="swatch" style="background:var(--slate)"></span>3B (mapping)<span class="n" id="donut-3b-pct">—</span></div>
            <div class="legend-row"><span class="swatch" style="background:var(--amber)"></span>24B (repair/answer)<span class="n" id="donut-24b-pct">—</span></div>
          </div>
        </div>
        <div class="chunk-summary" style="border-top:1px solid var(--line);margin-top:16px;padding-top:14px;">
          가벼운 추출은 3B(mapping), 보정·최종 종합은 24B(repairing/answering)로 라우팅
        </div>
      </div>
    </div>

    <div class="history">
      <div class="card-head" style="margin-bottom:12px;">
        <h2 style="font-size:15px;">질문 히스토리</h2>
        <button class="primary" id="download-submission-btn" disabled style="margin-left:auto;">submission.json 다운로드</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>ID</th><th>질문</th><th>카테고리</th><th>상태</th></tr></thead>
          <tbody id="history-body">
            <tr><td colspan="4" class="qtext" style="color:var(--ink-faint)">아직 실행한 질문이 없습니다</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>
</div>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Verify visually**

Open `web/index.html` directly in a browser (double-click, or `file://` URL — no server
needed yet since there's no `app.js` logic to run). Confirm: sidebar + 4 stat cards + upload
card + ask card + log card + donut card + history table all render without visual breakage,
in both light and dark OS theme (toggle your OS setting or devtools "Rendering > Emulate CSS
prefers-color-scheme").

- [ ] **Step 4: Commit**

```bash
git add web/index.html web/styles.css
git commit -m "feat(web): add dashboard HTML/CSS shell"
```

---

## Task 5: `app.js` — document + question-set upload

**Files:**
- Create: `web/app.js`

- [ ] **Step 1: Create `web/app.js` with upload logic**

```javascript
// web/app.js — talks to the FastAPI backend directly; no build step, no framework.

const API_BASE = "http://127.0.0.1:8000";

const state = {
  documentId: null,
  chunkCount: 0,
  questions: [], // [{question_id, question, category}]
  submission: null,
  stageTally: { "3B": 0, "24B": 0 },
  logFilter: "all",
};

function el(id) {
  return document.getElementById(id);
}

function showError(message) {
  const box = el("upload-error");
  box.textContent = message;
  box.style.display = "block";
}

function clearError() {
  el("upload-error").style.display = "none";
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function addFileRow(ext, name, size, stateLabel, chipClass) {
  const row = document.createElement("div");
  row.className = "file";
  row.innerHTML = `
    <div class="ext ${ext.toLowerCase() === "txt" ? "txt" : "yaml"}">${ext}</div>
    <div class="name">${name}</div>
    <div class="size mono">${size}</div>
    <div class="state"><span class="chip ${chipClass}">${stateLabel}</span></div>
  `;
  el("file-list").appendChild(row);
}

function maybeEnableRunButtons() {
  const ready = Boolean(state.documentId) && state.questions.length > 0;
  el("run-selected-btn").disabled = !ready;
  el("run-all-btn").disabled = !ready;
}

// ---------- document upload ----------

async function uploadDocument(file) {
  clearError();
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch(`${API_BASE}/documents`, { method: "POST", body: formData });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `업로드 실패 (${res.status})`);
    }
    const doc = await res.json();
    state.documentId = doc.document_id;
    state.chunkCount = doc.chunk_count;
    renderDocumentUploaded(doc, file);
  } catch (err) {
    showError(err.message);
  }
}

function renderDocumentUploaded(doc, file) {
  addFileRow(file.name.split(".").pop().toUpperCase(), file.name, formatBytes(file.size), "업로드 완료", "chip-ok");
  el("chunk-summary").style.display = "flex";
  el("chunk-summary-count").textContent = doc.chunk_count;
  el("stat-chunks").textContent = doc.chunk_count;
  el("stat-chunks-status").textContent = "업로드 완료";
  el("index-doc-label").textContent = `${doc.filename} · ${doc.chunk_count}개 청크`;
  el("index-bar").style.width = "100%";
  el("index-chunk-count").innerHTML = `<b>${doc.chunk_count}</b> / <b>${doc.chunk_count}</b>`;
  maybeEnableRunButtons();
}

// ---------- question set ----------

async function loadQuestions(file) {
  clearError();
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    const items = Array.isArray(data) ? data : data.questions;
    if (!Array.isArray(items)) {
      throw new Error("JSON은 질문 배열이거나 {questions: [...]} 형식이어야 합니다.");
    }
    const questions = items.map((item) => ({
      question_id: String(item.id ?? item.question_id ?? ""),
      question: String(item.question ?? item.text ?? ""),
      category: String(item.category ?? ""),
    }));
    if (questions.some((q) => !q.question_id || !q.question)) {
      throw new Error("모든 질문에 id와 question이 있어야 합니다.");
    }
    state.questions = questions;
    addFileRow("JSON", file.name, formatBytes(file.size), `${questions.length}문항`, "chip-cat");
    el("questions-count-label").textContent = `${questions.length}문항 로드됨`;
    el("stat-question-count").textContent = questions.length;
    populateQuestionPicker();
    maybeEnableRunButtons();
  } catch (err) {
    showError(`질문 파일을 읽을 수 없습니다: ${err.message}`);
  }
}

function populateQuestionPicker() {
  const picker = el("question-picker");
  picker.innerHTML = "";
  state.questions.forEach((q) => {
    const opt = document.createElement("option");
    opt.value = q.question_id;
    const preview = q.question.length > 40 ? `${q.question.slice(0, 40)}…` : q.question;
    opt.textContent = `${q.question_id} — ${preview}`;
    picker.appendChild(opt);
  });
  picker.disabled = false;
  updatePickedCategory();
}

function updatePickedCategory() {
  const picker = el("question-picker");
  const picked = state.questions.find((q) => q.question_id === picker.value);
  el("picked-category").textContent = picked && picked.category ? picked.category : "미분류";
}

// ---------- wiring (upload only for now) ----------

function initUploadHandlers() {
  el("doc-input").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadDocument(e.target.files[0]);
  });
  el("doc-dropzone").addEventListener("dragover", (e) => e.preventDefault());
  el("doc-dropzone").addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) uploadDocument(e.dataTransfer.files[0]);
  });

  el("questions-input").addEventListener("change", (e) => {
    if (e.target.files[0]) loadQuestions(e.target.files[0]);
  });

  el("question-picker").addEventListener("change", updatePickedCategory);
}

document.addEventListener("DOMContentLoaded", initUploadHandlers);
```

- [ ] **Step 2: Verify manually against the real backend**

Terminal 1 (from the repo root, with `.env` set up per the existing README):

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd web
python -m http.server 5500
```

Open `http://127.0.0.1:5500`, upload `week4/nationalparks_europe.txt` (from the parent
project directory — copy it into a scratch location first if the browser's file picker
can't reach outside the repo). Confirm: the upload card shows a file row with "업로드 완료",
the chunk-summary line appears with a real chunk count, the sidebar card fills its bar to
100%, and the stat row's "색인 청크" updates.

Convert the practice question set for this and later steps:

```bash
python -c "
import yaml, json
data = yaml.safe_load(open('../dev_questions.yaml', encoding='utf-8'))
items = data if isinstance(data, list) else data.get('questions', data)
out = [{'id': q.get('id', q.get('question_id')), 'question': q.get('question', q.get('text')), 'category': q.get('category', '')} for q in items]
json.dump(out, open('fixtures_dev_questions.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(len(out), 'questions written')
"
```

(Run this from `web/`, with `pyyaml` installed — `pip install pyyaml` if needed. Adjust the
source path if `dev_questions.yaml` isn't at `../../dev_questions.yaml` relative to `web/`;
check with `ls ../../dev_questions.yaml` first.)

Upload the resulting `web/fixtures_dev_questions.json` through the "질문 세트(JSON) 선택"
control. Confirm: a second file row appears, "질문 세트" stat updates to the real count
(21 for the practice set), and both "이 질문만 실행"/"전체 실행" buttons become enabled.

- [ ] **Step 3: Commit**

```bash
git add web/app.js
echo "web/fixtures_dev_questions.json" >> .gitignore
git add .gitignore
git commit -m "feat(web): wire document and question-set upload"
```

(The converted fixture is a local convenience file, not checked in — regenerate it with the
command above whenever needed. If the team wants it version-controlled for everyone's
convenience instead, drop the `.gitignore` line and `git add` the fixture directly.)

---

## Task 6: `app.js` — run + SSE progress

**Files:**
- Modify: `web/app.js`

- [ ] **Step 1: Add stage info, run, and SSE streaming**

Append to `web/app.js` (before the `initUploadHandlers`/`DOMContentLoaded` lines at the
bottom — move those to the very end after Step 1 of Task 7):

```javascript
// ---------- stage -> model badge (see spec's confirmed lookup table) ----------

const STAGE_INFO = {
  compiling: { label: "문서 컴파일 중", badge: null },
  mapping: { label: "근거 추출 중 (전수 매핑)", badge: "3B" },
  repairing: { label: "보정 매핑 중", badge: "24B" },
  answering: { label: "최종 답변 종합 중", badge: "24B" },
};

// ---------- run ----------

function resetLog() {
  el("log-list").innerHTML = "";
}

async function startRun(questions) {
  clearError();
  resetLog();
  try {
    const res = await fetch(`${API_BASE}/answer-jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: state.documentId, questions }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `실행 시작 실패 (${res.status})`);
    }
    const job = await res.json();
    streamProgress(job.job_id);
  } catch (err) {
    showError(err.message);
  }
}

function streamProgress(jobId) {
  const source = new EventSource(`${API_BASE}/answer-jobs/${jobId}/stream`);

  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (!data.stage) return; // final-state replay payload has no "stage" guarantee; skip
    appendLogRow(data);
    updateProgressStats(data);
  };

  source.addEventListener("done", async () => {
    source.close();
    try {
      const res = await fetch(`${API_BASE}/answer-jobs/${jobId}`);
      const job = await res.json();
      if (job.status === "completed") {
        renderResult(job.result);
      } else {
        showError(job.error || "실행이 실패했습니다.");
      }
    } catch (err) {
      showError(`결과를 불러오지 못했습니다: ${err.message}`);
    }
  });

  source.onerror = () => {
    showError("진행 로그 연결이 끊겼습니다. 새로고침 후 이어보기를 시도하세요 (진행 상황은 서버에 남아 있습니다).");
  };
}

function appendLogRow(data) {
  const info = STAGE_INFO[data.stage] || { label: data.stage, badge: null };
  const li = document.createElement("li");
  li.dataset.badge = info.badge || "";
  const badgeHtml = info.badge
    ? `<span class="chip ${info.badge === "3B" ? "chip-3b" : "chip-24b"}">${info.badge}</span>`
    : "";
  const failedLabel = data.failed ? `, 실패 ${data.failed}` : "";
  li.innerHTML = `
    <div class="dot active"></div>
    <div>
      <div class="step">${info.label} (${data.processed}/${data.total}${failedLabel})</div>
      <div class="meta">${badgeHtml}<span class="time mono">${new Date().toLocaleTimeString("ko-KR")}</span></div>
    </div>
  `;
  el("log-list").appendChild(li);
  applyLogFilter();
  el("log-list").scrollTop = el("log-list").scrollHeight;
}

function applyLogFilter() {
  document.querySelectorAll("#log-list li").forEach((li) => {
    const matches = state.logFilter === "all" || li.dataset.badge === state.logFilter;
    li.style.display = matches ? "flex" : "none";
  });
}

function updateProgressStats(data) {
  el("stat-progress").textContent = `${data.processed}/${data.total}`;
  const info = STAGE_INFO[data.stage];
  if (info && info.badge) {
    state.stageTally[info.badge] += 1;
    updateDonut();
  }
}

function updateDonut() {
  const total = state.stageTally["3B"] + state.stageTally["24B"];
  if (total === 0) return;
  const pct3b = Math.round((state.stageTally["3B"] / total) * 100);
  const pct24b = 100 - pct3b;
  el("donut").style.background = `conic-gradient(var(--slate) 0 ${pct3b}%, var(--amber) ${pct3b}% 100%)`;
  el("donut-3b-pct").textContent = `${pct3b}%`;
  el("donut-24b-pct").textContent = `${pct24b}%`;
}
```

- [ ] **Step 2: Wire the run buttons**

In `initUploadHandlers`, add (rename the function to `initHandlers` since it now does more
than upload wiring — update the `document.addEventListener("DOMContentLoaded", ...)` call
at the bottom to match):

```javascript
  el("run-selected-btn").addEventListener("click", () => {
    const picked = el("question-picker").value;
    const question = state.questions.find((q) => q.question_id === picked);
    if (question) startRun([question]);
  });

  el("run-all-btn").addEventListener("click", () => {
    startRun(state.questions);
  });

  document.querySelectorAll(".pillnav button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pillnav button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      state.logFilter = btn.dataset.filter;
      applyLogFilter();
    });
  });
```

`renderResult` doesn't exist yet — that's Task 7. The app will show a JS console error on
`done` until then; that's expected at this checkpoint.

- [ ] **Step 3: Verify manually**

With both servers still running (Task 5's Step 2), click "이 질문만 실행" on the selected
question. Confirm: the log card fills with rows (문서 컴파일 중 → 근거 추출 중 → possibly
보정 매핑 중 → 최종 답변 종합 중), each with the right 3B/24B badge, the donut updates, and
the pill filter buttons (전체/3B/24B) actually hide/show rows when clicked. It's fine that
the run ends with a console error about `renderResult` — that confirms the `done` event and
job fetch worked, which is what this task is responsible for.

- [ ] **Step 4: Commit**

```bash
git add web/app.js
git commit -m "feat(web): run questions and stream SSE progress"
```

---

## Task 7: `app.js` — result rendering, download, usage polling

**Files:**
- Modify: `web/app.js`

- [ ] **Step 1: Add result rendering and usage polling**

Append to `web/app.js`:

```javascript
// ---------- result rendering ----------

function renderResult(result) {
  state.submission = result.submission;
  el("download-submission-btn").disabled = false;

  if (result.usage) {
    const total = (result.usage.total_input_tokens || 0) + (result.usage.total_output_tokens || 0);
    el("stat-tokens").textContent = total.toLocaleString("en-US");
  }

  const tbody = el("history-body");
  if (tbody.dataset.seeded !== "true") {
    tbody.innerHTML = "";
    tbody.dataset.seeded = "true";
  }

  let lastAnswer = null;
  result.answers.forEach((answer) => {
    const question = state.questions.find((q) => q.question_id === answer.question_id);
    const failed = Boolean(answer.warnings && answer.warnings.length > 0);
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="qid mono">${answer.question_id}</td>
      <td class="qtext">${question ? question.question : ""}</td>
      <td><span class="chip chip-cat">${question && question.category ? question.category : "미분류"}</span></td>
      <td><span class="status-pill ${failed ? "" : "ok"}">${failed ? "실패" : "완료"}</span></td>
    `;
    tbody.appendChild(row);
    lastAnswer = { answer, question };
  });

  if (lastAnswer) renderAnswerCard(lastAnswer.answer, lastAnswer.question);
}

function renderAnswerCard(answer, question) {
  el("answer-block").style.display = "block";
  el("answer-text").textContent = answer.final_answer;
  const failed = Boolean(answer.warnings && answer.warnings.length > 0);
  el("answer-meta").innerHTML = `
    <span class="chip chip-cat">${question && question.category ? question.category : "미분류"}</span>
    ${failed ? `<span class="chip chip-err">${answer.warnings[0]}</span>` : `<span class="chip chip-ok">완료</span>`}
  `;
}

function downloadSubmission() {
  if (!state.submission) return;
  const blob = new Blob([JSON.stringify(state.submission, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "submission.json";
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- usage polling ----------
// NOTE: app/llm/broker.py passes the course broker's /usage response through
// verbatim -- its exact field names aren't confirmed by this plan. Run
// `curl http://127.0.0.1:8000/usage` once the .env broker key is configured
// (Task 7's manual verification step) and adjust formatUsageCost's field
// lookups below to match.

function formatUsageCost(usage) {
  const cost = usage.total_cost ?? usage.cost ?? usage.total_cost_usd;
  if (typeof cost === "number") return `$${cost.toFixed(2)}`;
  return "확인 필요"; // placeholder until the real field name is confirmed
}

async function pollUsage() {
  try {
    const res = await fetch(`${API_BASE}/usage`);
    if (!res.ok) return;
    const usage = await res.json();
    el("stat-cost").textContent = formatUsageCost(usage);
  } catch {
    // usage is best-effort; ignore transient failures
  }
}
```

- [ ] **Step 2: Wire the download button and start usage polling**

In the handlers function (renamed `initHandlers` in Task 6), add:

```javascript
  el("download-submission-btn").addEventListener("click", downloadSubmission);
```

Change the bottom of the file from (this is what Task 6 Step 2 left it as):

```javascript
document.addEventListener("DOMContentLoaded", initHandlers);
```

to:

```javascript
document.addEventListener("DOMContentLoaded", () => {
  initHandlers();
  pollUsage();
  setInterval(pollUsage, 5000);
});
```

- [ ] **Step 3: Verify manually, end to end**

Repeat Task 6 Step 3's run. Confirm this time: no console error, the answer card fills in
with real text, a new row appears in the history table with the right category/status, the
download button becomes enabled, and clicking it saves a `submission.json` file whose
`answers` array matches what the answer card showed.

Then click "전체 실행" with all 21 practice questions loaded. Confirm: the log accumulates
many rows across all 21 questions' stages, the donut's 3B/24B split moves as more mapping
vs. repairing/answering events arrive, and — once the whole job finishes — the history table
gets one row per question and the download produces a `submission.json` with 21 answers.
(This is expected to take a while and cost real broker tokens — this is the same pipeline
the team already validated, just triggered from the new UI.)

Confirm the `/usage` field-name placeholder from Step 1: run
`curl http://127.0.0.1:8000/usage` in a terminal, note the real field name for total cost,
and update `formatUsageCost` in `web/app.js` to read it directly (remove the
`"확인 필요"` fallback once confirmed).

- [ ] **Step 4: Commit**

```bash
git add web/app.js
git commit -m "feat(web): render results, download submission, poll usage"
```

---

## Task 8: Read-only pipeline parameter panel

Added after professor feedback: chatbot UI polish scores less than expected; a simple
display of the pipeline's current parameters (chunk size, batch sizes, models, etc.) is
more valuable. Scoped to **read-only** — actually changing chunk size and re-running is a
bigger change that touches `app/chunking/` and `app/pipeline.py`'s signatures (the pipeline
pair's files), out of scope for today.

**Files:**
- Modify: `app/api/main.py` (add one new route)
- Test: `tests/test_api.py`
- Modify: `web/index.html` (one new card)
- Modify: `web/app.js` (one fetch + render function)

- [ ] **Step 1: Write the failing backend test**

Add to `tests/test_api.py`:

```python
class TestPipelineSettings:
    def test_pipeline_settings_excludes_secrets(self, client):
        response = client.get("/pipeline-settings")
        assert response.status_code == 200
        data = response.json()
        assert "mistral_api_key" not in data
        assert "chunk_target_tokens" in data
        assert "mapper_model" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py::TestPipelineSettings -v`
Expected: FAIL — 404, route doesn't exist.

- [ ] **Step 3: Add the route**

Add anywhere after the `/usage` route in `app/api/main.py` (locate `async def broker_usage():`
by content, not line number — see the caveat under Task 2):

```python
@app.get("/pipeline-settings")
async def pipeline_settings():
    """Expose the current non-secret pipeline configuration for display only."""
    settings = get_settings()
    return {
        "mapper_model": settings.mapper_model,
        "answer_model": settings.answer_model,
        "verifier_model": settings.verifier_model,
        "planner_model": settings.planner_model,
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "question_batch_size": settings.question_batch_size,
        "record_batch_size": settings.record_batch_size,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "max_retries": settings.max_retries,
        "pipeline_mode": settings.pipeline_mode.value,
        "llm_temperature": settings.llm_temperature,
        "llm_top_p": settings.llm_top_p,
        "llm_seed": settings.llm_seed,
        "team_name": settings.team_name,
    }
```

Deliberately excludes `mistral_api_key`, `llm_base_url`, `ollama_base_url`, and `data_dir` —
either secret or not useful to show.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py::TestPipelineSettings -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `python -m pytest tests/ -v -m "not live"`
Expected: all PASS.

- [ ] **Step 6: Commit the backend change**

```bash
git add app/api/main.py tests/test_api.py
git commit -m "feat(api): add read-only pipeline-settings endpoint"
```

- [ ] **Step 7: Add the display card to `web/index.html`**

Insert this new card right before the `<div class="history">` block:

```html
<div class="card" style="margin-bottom:14px;">
  <div class="card-head"><h2>파이프라인 파라미터</h2><span class="sub">읽기전용 · .env 기준</span></div>
  <div class="stats" id="settings-grid" style="margin-bottom:0;">
    <div class="stat-card"><div class="l">불러오는 중…</div></div>
  </div>
</div>
```

- [ ] **Step 8: Render it from `web/app.js`**

Append:

```javascript
// ---------- pipeline parameters (read-only) ----------

const SETTINGS_LABELS = {
  mapper_model: "매핑 모델 (3B)",
  answer_model: "답변/보정 모델 (24B)",
  verifier_model: "검증 모델",
  planner_model: "플래너 모델",
  chunk_target_tokens: "청크 목표 토큰",
  chunk_overlap_tokens: "청크 오버랩 토큰",
  question_batch_size: "질문 배치 크기",
  record_batch_size: "레코드 배치 크기",
  max_concurrent_requests: "최대 동시 요청",
  request_timeout_seconds: "요청 타임아웃(초)",
  max_retries: "최대 재시도",
  pipeline_mode: "파이프라인 모드",
  llm_temperature: "temperature",
  llm_top_p: "top_p",
  llm_seed: "seed",
  team_name: "팀 이름",
};

async function loadPipelineSettings() {
  try {
    const res = await fetch(`${API_BASE}/pipeline-settings`);
    if (!res.ok) return;
    const settings = await res.json();
    const grid = el("settings-grid");
    grid.innerHTML = "";
    Object.entries(settings).forEach(([key, value]) => {
      const card = document.createElement("div");
      card.className = "stat-card";
      card.innerHTML = `
        <div class="l">${SETTINGS_LABELS[key] || key}</div>
        <div class="n" style="font-size:15px;">${value}</div>
      `;
      grid.appendChild(card);
    });
  } catch {
    // best-effort display only
  }
}
```

Call it once at startup — in the `DOMContentLoaded` handler from Task 7 Step 2, add
`loadPipelineSettings();` alongside `initHandlers(); pollUsage(); ...`.

- [ ] **Step 9: Verify manually**

With both servers running, reload `http://127.0.0.1:5500`. Confirm the new card shows real
values pulled from the repo's `.env` (chunk sizes, batch sizes, both model names, etc.), and
that no secret (`mistral_api_key`) appears anywhere in the page or in the Network tab
response body for `/pipeline-settings`.

- [ ] **Step 10: Commit**

```bash
git add web/index.html web/app.js
git commit -m "feat(web): display read-only pipeline parameters"
```

---

## Task 9: Final check

**Files:** none (verification only)

- [ ] **Step 1: Run the full backend test suite one more time**

Run: `python -m pytest tests/ -v -m "not live"`
Expected: all PASS.

- [ ] **Step 2: Run ruff**

Run: `python -m ruff check app tests`
Expected: no new issues introduced by Tasks 1-3 (pre-existing issues elsewhere in the repo,
if any, are not this plan's responsibility to fix).

- [ ] **Step 3: Confirm `ui/streamlit_app.py` still works untouched**

Run: `python -m streamlit run ui/streamlit_app.py --server.port 8501` and confirm it starts
without errors — this plan must not have broken the existing demo path.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin feature/live-dashboard
```

(Ask before opening a PR against `main` — that's a decision for the whole team, not this
plan.)

---

## Plan Review Loop

Before execution, dispatch a plan-document-reviewer (or, if unavailable in this environment,
a `general-purpose` agent with reviewer instructions, as was done for the spec) against this
plan and the spec at `docs/superpowers/specs/2026-08-04-live-dashboard-design.md`. Fix any
issues found and re-dispatch until approved, max 3 iterations, then surface to the human.
