# FULLSCAN-QA V3

Exhaustive long-document question answering for the DASU capstone, without RAG.

The runtime does not use embeddings, vector search, BM25, reranking, or top-k
context selection. Every compiled record is assigned a terminal result for each
unresolved question. Python performs counts, maxima, minima, and absence proofs.

## Architecture

V3 has one production path: `ADAPTIVE_HIERARCHICAL`.

```text
PDF/TXT
  -> page-preserving parser
  -> question-independent Document Compiler
       -> generic repeated-entity registry with integrity checks
       -> boilerplate-cycle chapter segmentation when fact anchors are lost
       -> deterministic number/label binding for mixed fact-card layouts
       -> fields named by the document's own wording, one naming authority
       -> wording variants of one metric folded into a single field
  -> per-record structured profiling (only when the registry has holes)
       -> induced metric vocabulary, then one cheap call per record
       -> every fact kept must be quoted verbatim from its own record
  -> question-to-field binding against the discovered field list
       -> matched by meaning, bounded to fields the document actually has
       -> flattened multi-page table reconstruction and ranked-row validation
       -> body-caption cross-validation for interleaved Contents lists
       -> consecutive fallback segments for arbitrary Markdown
       -> supplementary front/back-matter records
  -> deterministic question compiler with composable operation plans
  -> Python structured executor for FILTER, GROUP_BY, COUNT, LIST, ARGMAX,
     ARGMIN, JOIN, comparison, and date math
  -> exhaustive batched mapper over every record for unresolved questions
  -> category-specific reducer
       -> absence coverage matrix
       -> Python count/argmax
       -> claim comparison
       -> hierarchical evidence packet
  -> Mistral Small final synthesis
  -> bounded stronger-model repair only for empty or position-poor evidence
  -> one slot-refinement pass only when a requested field is objectively missing
  -> incremental grader-compatible submission.json
```

### Strategy Routing

| Question class | Execution |
|---|---|
| Aggregation / superlative | Typed facts and Python Reduce; exhaustive mapped facts as fallback |
| Flattened tables / Contents | Validated row schemas or caption-cross-checked identifier indexes |
| Absence | Candidate-topic x every-record coverage matrix |
| Contradiction | Deterministic comparison when possible; otherwise exhaustive claim evidence |
| Cross-section | Exhaustive map and complete evidence synthesis |
| Global synthesis | Record-level map followed by hierarchical evidence Reduce |
| Needle lookup | Exhaustive map followed by a concise grounded answer |

The final answer model never replaces a correct deterministic result. There is
no general verifier that can rewrite a count, maximum, or absence verdict.

## Reliability Properties

- Every page belongs to a catalog, segment, or supplementary mapping record.
- Repeated-entity registries are trusted only when fact-section, title, and
  contents counts agree; otherwise the exhaustive fallback remains available.
- Inline numbered profiles are accepted only when ordinals, titles, countries,
  and labelled fact cards pass completeness checks.
- A structured count or maximum is withheld unless the target field is bound
  for every record; partial coverage defers to the exhaustive mapper rather
  than answering from the rows that happened to parse.
- Records are segmented by recurring chapter furniture when fact-section
  anchors go missing, so the record count is calibrated by the document
  itself rather than by domain vocabulary.
- Profiling is skipped entirely when the compiler already bound a fact card
  for every record, so well-parsed documents cost no extra tokens.
- No field name, question phrasing, or subject vocabulary is written into the
  code. Fields are named by the document, and a question reaches one only by
  being matched against the fields that document actually reports.
- Contents counts are trusted only when their identifiers are confirmed by
  source-literal body captions; ambiguous layouts use a bounded cached fallback.
- Stored documents are automatically recompiled when the compiler contract changes.
- Every unresolved question/record pair receives `evidence_found`,
  `no_evidence`, `uncertain`, `parse_failed`, or `llm_failed`.
- Positive evidence must contain a source-literal quote.
- A quote shortened with an ellipsis is accepted only when a literal fragment
  of at least 30 characters locates the complete source sentence.
- Missing result rows and missing absence topics are detected automatically.
- Mapper JSON gets one bounded schema-repair attempt.
- High-risk absence, contradiction, and needle questions are mapped in isolated
  question state. Weak global synthesis receives a small early/middle/late repair set.
- Mapper results are cached by record content, plan, model, and prompt contract.

## Models

Only the course broker models are used:

| Stage | Broker model |
|---|---|
| Ambiguous structure fallback | `mistral-small-3-2` |
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
RECORD_BATCH_MAX_CHARACTERS=18000
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
- generic repeated-entity detection, registry integrity, and fictional countries;
- composed FILTER -> ARGMAX and JOIN -> date-difference execution;
- inline profile completeness, repeated headers, ranked table rows, and unranked exclusions;
- interleaved Contents reconstruction through body-caption cross-validation;
- non-profile needle subjects and generic first-recorded events;
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
data/cache/v3/structure/*.json bounded ambiguous-layout cache
data/runs/*_submission.json    incremental grader file
data/runs/<run-id>/*.json      plans, map results, diagnostics
data/runs/<run-id>.json        completed PipelineRun
```

The practice document compiles into 60 park records plus supplementary
front/index records. Other repeated-profile documents use the same registry
compiler without requiring park-specific headings or real-world country names.
Structured questions are resolved from the trusted catalog without LLM
arithmetic; unresolved questions still scan every compiled record.
