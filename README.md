# FULLSCAN-QA V3

Exhaustive long-document question answering for the DASU capstone, without RAG.

The runtime does not use embeddings, vector search, BM25, reranking, or top-k
context selection. Every compiled record is assigned a terminal result for each
unresolved question. Python performs counts, maxima, minima, and absence proofs.

## Architecture

V3 has one production path: `ADAPTIVE_HIERARCHICAL_V3`.

```text
PDF/TXT
  -> page-preserving parser
  -> question-independent Document Compiler
       -> closed repeated-record catalog when structure is detected
       -> deterministic number/label binding for numeric cards
       -> consecutive fallback segments for arbitrary Markdown
       -> supplementary front/back-matter records
  -> deterministic question compiler
  -> Python structured executor when typed facts are sufficient
  -> exhaustive batched mapper over every record for unresolved questions
  -> category-specific reducer
       -> absence coverage matrix
       -> Python count/argmax
       -> claim comparison
       -> hierarchical evidence packet
  -> Mistral Small final synthesis
  -> one slot-refinement pass only when a requested field is objectively missing
  -> incremental grader-compatible submission.json
```

### Strategy Routing

| Question class | Execution |
|---|---|
| Aggregation / superlative | Typed facts and Python Reduce; exhaustive mapped facts as fallback |
| Absence | Candidate-topic x every-record coverage matrix |
| Contradiction | Deterministic comparison when possible; otherwise exhaustive claim evidence |
| Cross-section | Exhaustive map and complete evidence synthesis |
| Global synthesis | Record-level map followed by hierarchical evidence Reduce |
| Needle lookup | Exhaustive map followed by a concise grounded answer |

The final answer model never replaces a correct deterministic result. There is
no general verifier that can rewrite a count, maximum, or absence verdict.

## Reliability Properties

- Every page belongs to a catalog, segment, or supplementary mapping record.
- Every unresolved question/record pair receives `evidence_found`,
  `no_evidence`, `uncertain`, `parse_failed`, or `llm_failed`.
- Positive evidence must contain a source-literal quote.
- A quote shortened with an ellipsis is accepted only when a literal fragment
  of at least 30 characters locates the complete source sentence.
- Missing result rows and missing absence topics are detected automatically.
- Mapper JSON gets one bounded schema-repair attempt.
- Technical or coverage repair is batched across affected questions; it is not
  repeated independently for every question.
- Mapper results are cached by record content, plan, model, and prompt contract.

## Models

Only the course broker models are used:

| Stage | Broker model |
|---|---|
| Exhaustive evidence mapping | `ministral-3b-2512` |
| Final synthesis / bounded repair | `mistral-small-3-2` |

The deterministic compiler, router, reducers, coverage checks, and submission
writer do not call an LLM.

## Install

```powershell
cd C:\Users\user\Desktop\THU_Project\fullscan-qa
pip install -e ".[dev]"
```

Create `.env` from `.env.example` and set the course group key. Never commit
the real `.env` file.

Important values for the course broker:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://80.151.131.52:9839/v1
MISTRAL_API_KEY=<group-key>
MAPPER_MODEL=ministral-3b-2512
ANSWER_MODEL=mistral-small-3-2
MAX_CONCURRENT_REQUESTS=3
REQUEST_TIMEOUT_SECONDS=300
MAX_RETRIES=1
QUESTION_BATCH_SIZE=8
RECORD_BATCH_SIZE=3
PIPELINE_MODE=ADAPTIVE_HIERARCHICAL
TEAM_NAME=C
```

## Run

Terminal 1:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
python -m streamlit run ui/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501`, upload the supplied machine-readable `.txt`
document, upload `questions.json`, and process the questions. The PDF path is
supported but the capstone instructions recommend TXT.

The CLI accepts either format:

```powershell
python scripts/run_pipeline.py C:\path\document.txt `
  -q "What did the Icehotel start out as, and in what year?" `
  -o run.json
```

## API

- `GET /health`
- `GET /health/llm`
- `GET /usage`
- `POST /documents`
- `GET /documents/{document_id}`
- `POST /answer`
- `POST /answer-jobs`
- `GET /answer-jobs/{job_id}`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/submission`

Long-running UI requests use background jobs. Progress stages are `compiling`,
`mapping`, `repairing`, and `answering`.

## Question Input

Only question IDs, text, and optional categories belong in `questions.json`:

```json
[
  {
    "id": "d01",
    "category": "aggregation",
    "question": "How many ...?"
  }
]
```

Do not provide grading `key_points` or `forbidden` content to the answering
system. Those fields are evaluation references, not model input.

## Submission

Each run incrementally writes:

```json
{
  "team": "C",
  "notes": "...",
  "answers": [
    {"id": "d01", "answer": "...", "evidence": ["..."]}
  ]
}
```

The partial file remains valid if a later question fails or a deadline is hit.

## Verification

```powershell
python -m ruff check app tests scripts
python -m pytest tests -q -m "not live"
python -m compileall -q app ui scripts
```

V3 regression tests cover:

- numeric value/label binding, million scaling, and BC dates;
- complete generic-page segmentation;
- exhaustive mapper pair coverage and exact-quote validation;
- ellipsis-to-source quote restoration;
- no-LLM structured execution;
- end-to-end map, Reduce, answer, coverage, and submission behavior.

## Artifacts

```text
data/parsed/*_parsed.json       page-preserving parse
data/parsed/*_compiled_v3.json V3 catalog and typed facts
data/cache/v3/*.json           validated mapper cache
data/runs/*_submission.json    incremental grader file
data/runs/<run-id>/*.json      plans, map results, diagnostics
data/runs/<run-id>.json        completed PipelineRun
```

The practice document compiles into 60 park records plus 3 supplementary
front/index records. Its structured questions are resolved from the closed
catalog without LLM arithmetic. The remaining questions require two exhaustive
mapper passes in the default batching layout, followed by at most seven final
answer calls unless a bounded repair pass is triggered.
