# AGENTS.md — FULLSCAN-QA Coding Agent Guidelines

## Project Goals
FULLSCAN-QA is an operator-aware exhaustive large-document QA system.
It answers complex questions (aggregation, superlative, absence, counts)
by processing EVERY document record — not by retrieving top-k.

## Hard Rules

### No RAG — EVER
Do NOT add:
- Vector databases (FAISS, Chroma, Pinecone, Weaviate)
- Embeddings of any kind
- Semantic similarity search
- BM25 or TF-IDF retrieval
- Top-k chunk selection
- Reranking
- Query-dependent chunk filtering
- LangChain/LlamaIndex retrievers

### Allowed Models
Only these model families may be used:
- Mistral Small 3.2 (e.g., `mistral-small-2506`)
- Ministral 3 8B (e.g., `ministral-8b-2512`)

No Gemini, OpenAI, Claude, Cohere, Llama, embedding models, or OCR models.

### Architecture Boundaries
- LLMs extract structured evidence from text
- Python performs ALL deterministic computation (count, max, min, sum, etc.)
- LLMs generate and verify natural-language answers
- Never ask an LLM to count a list or calculate a maximum

### Deterministic Reducers
Do NOT replace the Python-based deterministic reducers with LLM calls.
COUNT, ARGMAX, ARGMIN, SUM, AVERAGE, DIFFERENCE, PERCENT_CHANGE,
FILTER_LIST, FILTER_COUNT_LIST, COMPARE, TEMPORAL, and ABSENCE
must all be computed by Python.

### Failed Chunks
Do NOT skip or silently ignore chunks that fail processing.
Every chunk must have a terminal status: evidence_found, no_evidence,
uncertain, parse_failed, or llm_failed.

### Secrets
Do NOT hardcode API keys. Use environment variables only.
Never commit `.env` files.

## Testing Commands
```bash
pytest tests/ -v                    # All tests
pytest tests/ -v --cov=app          # With coverage
pytest tests/ -v -k "not live"      # Skip live API tests
```

## Formatting
```bash
ruff check app/ tests/ --fix
ruff format app/ tests/
```

## Prohibited Dependencies
- LangChain
- LlamaIndex
- Any vector store library
- Any embedding library
- Any workflow orchestration framework
- Any heavy distributed computing framework

## File Organization
- `app/api/` — FastAPI endpoints
- `app/llm/` — LLM client abstraction and model routing
- `app/parsing/` — page-preserving PDF/TXT parsing
- `app/reduction/` — shared normalization and deterministic operators
- `app/storage/` — compiled-document and run artifact persistence
- `app/v3/compiler.py` — generic repeated-entity document compiler
- `app/v3/question_compiler.py` — composable question operation plans
- `app/v3/structured_executor.py` — deterministic Python execution
- `app/v3/exhaustive_mapper.py` — exhaustive evidence extraction
- `app/v3/reducers.py` — coverage and evidence reduction
- `app/v3/answerer.py` — concise evidence-grounded synthesis
- `app/pipeline.py` — sole production pipeline orchestration
- `tests/` — regression and integration tests
- `scripts/` — CLI tools
- `ui/` — Streamlit demo
