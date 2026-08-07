# Running the dashboard against both Sandwich and Croissant

This submission ships two backend branches, each with its own pipeline but
the same dashboard frontend so you can switch between them live from one
screen:

| Backend  | Branch                                 | Pipeline                          | Port |
|----------|-----------------------------------------|------------------------------------|------|
| Sandwich | `feature/dashboard-on-yyk-sandwich`     | `hdr2025/pipeline_g14.py`          | 8003 |
| Croissant| `feature/dashboard-on-pyoseon-croissant`| `app/pipeline.py` (V3, pyoseon's)  | 8004 |

Both branches contain an identical `web/` folder, so you only need to serve
it once -- but you need **two separate checkouts** running at the same time
(one backend process per branch), since they're different codebases.

## Setup

```bash
# 1. Two working copies, one per branch
git clone <repo-url> sandwich-checkout
cd sandwich-checkout && git checkout feature/dashboard-on-yyk-sandwich
cd ..
git clone <repo-url> croissant-checkout
cd croissant-checkout && git checkout feature/dashboard-on-pyoseon-croissant
cd ..

# 2. Each needs its own .env with MISTRAL_API_KEY / LLM_BASE_URL set
#    (copy .env.example -> .env in both and fill in the broker key)

# 3. Install dependencies in each (same pyproject.toml in both)
pip install -e sandwich-checkout
pip install -e croissant-checkout
pip install openai pyyaml   # extra deps pipeline_g14.py needs, Sandwich only

# 4. Run both backends
cd sandwich-checkout && uvicorn app.api.main:app --host 127.0.0.1 --port 8003 &
cd ../croissant-checkout && uvicorn app.api.main:app --host 127.0.0.1 --port 8004 &

# 5. Serve the dashboard (from either checkout, they're identical)
cd sandwich-checkout/web && python -m http.server 5500
```

Open `http://127.0.0.1:5500`, and use the **Model backend** dropdown
(top of the right column) to switch between Sandwich and Croissant.
Switching resets the current session -- re-upload the document after
changing backends.

## Known differences between the two

- **Sandwich** only answers `.txt` documents (its pipeline reads the file as
  plain text; a `.pdf` will error on the answer step, though the file can
  still be *viewed* in the source panel). Evidence quotes show a raw
  provenance block rather than a clean sentence, and page numbers are rarely
  attached, so highlighting mostly falls back to "page n/a".
- **Croissant** supports both `.pdf` and `.txt` for answering and viewing,
  same as the main pipeline. Evidence/page pairing is patched only through
  `pipeline.py`, not every `structured_executor.py` function, so some
  answers still show "page n/a" instead of a highlighted jump.
- Both track real per-model token usage and cost live via the same
  `/usage` and `/pipeline-settings` endpoints.

See `web/INTEGRATION.md` in each branch for the exact code-level patches, if
you need to connect a third pipeline the same way.
