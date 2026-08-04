"""
Hybrid Map-Reduce + Refine pipeline for the Long-Context Challenge.

Idea:
- MAP: exhaustively extract structured facts from every chunk (never skips a
  chunk, unlike RAG top-k retrieval) -> needed for aggregation/superlative/
  cross-section/absence/contradiction questions.
- REFINE: sequentially merge each chunk's facts into ONE running fact store,
  seeing only the store + one new chunk at a time -> keeps context small.
- ANSWER: runs separately, against the final (small) fact store, so you can
  ask many questions without re-processing the raw document each time.

Cost split (models are metered separately, per broker rules):
- MAP runs once per chunk -> many calls -> use the cheap model (ministral-3b-2512).
- REFINE and ANSWER need more careful reasoning but run less often
  (refine: once per chunk; answer: once per question) -> use mistral-small-3-2.

Requires:
    pip install openai pyyaml --break-system-packages
    export BROKER_API_KEY=your_group_key
"""

import os
import re
import json
import time
from openai import OpenAI

# ---------------------------------------------------------------- CONFIG

BASE_URL = "http://80.151.131.52:9839/v1"

MODEL_MAP = "ministral-3b-2512"      # cheap, runs once per chunk
MODEL_REFINE = "mistral-small-3-2"   # runs once per chunk, needs more care
MODEL_ANSWER = "mistral-small-3-2"   # runs once per question, graded output

_client = None

# Straight and typographic quotes. Keys pasted out of a PDF or a chat app often
# arrive wrapped in the typographic ones, which the shell keeps as part of the
# value and httpx then refuses to put in an ASCII header.
_QUOTE_CHARS = "\"'`‘’“”«»"


def _clean_api_key(raw: str) -> str:
    """Strip stray whitespace/quotes off a pasted key and reject non-ASCII ones."""
    key = raw.strip().strip(_QUOTE_CHARS).strip()
    try:
        key.encode("ascii")
    except UnicodeEncodeError as e:
        bad = key[e.start]
        raise RuntimeError(
            f"BROKER_API_KEY contains a non-ASCII character {bad!r} "
            f"(U+{ord(bad):04X}) at position {e.start}. It was most likely copied "
            "from a PDF or a chat app that rewrites quotes and dashes. Re-enter "
            "the key by typing the quotes yourself:\n"
            "  export BROKER_API_KEY=your_group_key"
        ) from None
    return key


def get_client() -> OpenAI:
    """
    Build the broker client on first use, not at import time.

    Importing this module must stay side-effect free: in a notebook you often
    want to import first and set the key in a later cell, and the OpenAI SDK's
    own error message names OPENAI_API_KEY, which is the wrong variable here.
    """
    global _client
    if _client is None:
        key = os.environ.get("BROKER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "BROKER_API_KEY is not set.\n"
                "  shell:    export BROKER_API_KEY=your_group_key\n"
                "  notebook: import os; os.environ['BROKER_API_KEY'] = 'your_group_key'"
            )
        # The SDK defaults to a 600s read timeout AND 2 internal retries, so one
        # stalled connection costs up to 30 minutes before it raises -- and our
        # own backoff loop would then retry it up to four more times. That is
        # what made a single question hang for 15+ minutes. Fail fast here and
        # let call_llm_raw's backoff own the retrying.
        _client = OpenAI(
            base_url=BASE_URL,
            api_key=_clean_api_key(key),
            # 150s was an overcorrection: a reader enumerating a theme that
            # runs through its whole slice is legitimately still generating at
            # that point, so it timed out, retried, and did the same slow thing
            # again. 300s covers the honest worst case without hiding a stall.
            timeout=float(os.environ.get("BROKER_TIMEOUT", "300")),
            max_retries=0,
        )
    return _client


# Set to False the first time the broker rejects response_format, so we stop
# asking for JSON mode on every later call.
_JSON_MODE_SUPPORTED = None


def call_llm_raw(prompt: str, model: str, json_mode: bool = False,
                 temperature: float | None = None) -> tuple[str, str]:
    """
    Send one prompt to the course broker.

    Returns (text, finish_reason). finish_reason == "length" means the model
    ran out of output tokens mid-sentence, which is the one failure that
    retrying verbatim will never fix -- the caller has to shrink the input.

    temperature=0 makes a run reproducible. Left unset, the broker samples at
    its own default and the same pipeline over the same data scores differently
    every time -- roughly +/-7 points on this task, which is larger than most
    of the improvements worth measuring.
    """
    global _JSON_MODE_SUPPORTED

    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    sampling = {} if temperature is None else {"temperature": temperature}

    def post(**extra):
        """One request, retried on transient broker errors with backoff."""
        last = None
        options = {**sampling, **extra}
        for attempt in range(4):
            try:
                return client.chat.completions.create(
                    model=model, messages=messages, **options
                )
            except Exception as e:
                last = e
                # A rejected response_format is a permanent 4xx, not transient:
                # let the caller fall back to a plain call immediately.
                if extra:
                    raise
                # Other permanent 4xx (bad key, oversized prompt) will fail the
                # same way every time -- don't burn 7s of backoff on them.
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                # Rate limits and hiccups get more likely once callers run
                # several chunks concurrently -- back off instead of dying.
                if attempt == 3:
                    break
                wait = 2 ** attempt
                print(f"  ! broker call failed ({type(e).__name__}); retrying in {wait}s")
                time.sleep(wait)
        raise last

    use_json_mode = json_mode and _JSON_MODE_SUPPORTED is not False
    try:
        if use_json_mode:
            response = post(response_format={"type": "json_object"})
            _JSON_MODE_SUPPORTED = True
        else:
            response = post()
    except Exception as e:
        if sampling and _rejects_parameter(e, "temperature"):
            # Broker doesn't accept temperature -- drop it rather than failing
            # every call. Runs stay non-reproducible, which is worth a warning.
            print("  ! broker rejected temperature; falling back to its default "
                  "(runs will not be reproducible)")
            sampling.clear()
            return call_llm_raw(prompt, model, json_mode=json_mode)
        if not use_json_mode:
            raise
        # Broker or model doesn't accept response_format -- remember and retry plain.
        _JSON_MODE_SUPPORTED = False
        response = post()

    choice = response.choices[0]
    return choice.message.content or "", (choice.finish_reason or "")


def _rejects_parameter(error: Exception, name: str) -> bool:
    """True when a 4xx names this parameter as the thing it did not like."""
    status = getattr(error, "status_code", None)
    if status is None or not (400 <= status < 500):
        return False
    return name in str(error).lower()


def call_llm(prompt: str, model: str, temperature: float | None = None) -> str:
    """Send one prompt to the course broker and return the raw text response."""
    text, _ = call_llm_raw(prompt, model, temperature=temperature)
    return text


def preprocess_document(raw_text: str) -> str:
    """
    Some source files store newlines as the literal two-character sequence
    '\\n' instead of real newline bytes (an artifact of how the PDF was
    converted to text). If left alone, these glue onto adjacent words when
    we split on whitespace. Turn them into real spaces here.

    Page markers like '<!-- page-start-marker-12 -->' are left in place --
    they're useful signal for cross-section questions (which page is this
    fact on?), not noise to strip.
    """
    return raw_text.replace("\\n", " ")


def chunk_document(text: str, words_per_chunk: int = 1500) -> list[str]:
    """Split document into chunks of ~words_per_chunk words, no overlap."""
    words = text.split()
    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


class LLMJSONError(ValueError):
    """The model could not be coaxed into producing parseable JSON."""

    def __init__(self, message: str, raw: str = "", truncated: bool = False):
        super().__init__(message)
        self.raw = raw
        self.truncated = truncated


_FENCED_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_block(raw: str) -> str:
    """
    Pull the JSON object out of a model response that may be wrapped in code
    fences and/or padded with prose. Tolerates an unterminated fence, which is
    what a truncated response looks like.
    """
    text = (raw or "").strip()

    fenced = _FENCED_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        return text
    end = text.rfind("}")
    return text[start:end + 1].strip() if end > start else text[start:].strip()


def safe_json_parse(raw: str) -> dict:
    """Strip code fences etc. and parse JSON, raising a clear error if it fails."""
    cleaned = extract_json_block(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM output as JSON:\n{raw}") from e


REPAIR_PROMPT = """The text below was supposed to be one valid JSON object, but it is malformed.
A JSON parser reported: {error}

Fix ONLY the syntax: mismatched or missing brackets, braces, quotes and commas.
Do not add, drop, rename, reorder or reword any data. Keep every value exactly
as written.

Output the corrected JSON object only -- no prose, no code fences.

BROKEN JSON:
{broken}
"""


def call_llm_json(prompt: str, model: str, *, repair_model: str | None = None,
                  max_attempts: int = 3, label: str = "",
                  temperature: float | None = 0.0) -> dict:
    """
    Call the model and return parsed JSON, tolerating the two ways small models
    fail at it:

    - malformed syntax (unbalanced brackets, stray commas) -> feed the broken
      text back with the parser's error message and ask for a syntax-only fix,
      using the stronger `repair_model`;
    - a truncated response (finish_reason == "length") -> unfixable here, so
      retry from scratch and finally raise with truncated=True so the caller
      can split its input.
    """
    repair_model = repair_model or model
    tag = f"{label}: " if label else ""
    broken: str | None = None
    last_error = "unknown error"
    truncated = False
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        if broken is None:
            raw, finish = call_llm_raw(prompt, model, json_mode=True,
                                       temperature=temperature)
        else:
            raw, finish = call_llm_raw(
                REPAIR_PROMPT.format(error=last_error, broken=broken),
                repair_model,
                json_mode=True,
                temperature=temperature,
            )
        last_raw = raw

        if finish == "length":
            truncated = True
            last_error = "response was cut off at the output-token limit"
            broken = None  # repairing a truncated body just loses data
            print(f"  ! {tag}{last_error} (attempt {attempt}/{max_attempts})")
            continue

        try:
            parsed = safe_json_parse(raw)
        except ValueError as e:
            truncated = False
            last_error = str(e.__cause__ or e)
            broken = extract_json_block(raw)
            print(f"  ! {tag}invalid JSON (attempt {attempt}/{max_attempts}): {last_error}")
            time.sleep(1)
            continue

        if attempt > 1:
            print(f"  * {tag}recovered on attempt {attempt}")
        return parsed

    raise LLMJSONError(
        f"{tag}gave up after {max_attempts} attempts -- {last_error}",
        raw=last_raw,
        truncated=truncated,
    )


_failure_count = 0


def _dump_failure(name: str, raw: str, directory: str = "failed_llm_output") -> str:
    """Keep the unparseable response on disk so a failure can be inspected later."""
    global _failure_count
    _failure_count += 1
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{_failure_count:02d}_{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw)
    return path


def empty_extraction(note: str | None = None) -> dict:
    """A schema-shaped extraction carrying no facts."""
    return {"entities": [], "superlatives": [], "topics_mentioned": [], "notes": note}


def merge_extractions(parts: list[dict]) -> dict:
    """
    Concatenate extractions locally (no LLM). Entity de-duplication is left to
    the REFINE step, which is what it exists for -- this only has to be lossless.
    """
    merged = empty_extraction()
    notes = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        merged["entities"].extend(part.get("entities") or [])
        merged["superlatives"].extend(part.get("superlatives") or [])
        for topic in part.get("topics_mentioned") or []:
            if topic not in merged["topics_mentioned"]:
                merged["topics_mentioned"].append(topic)
        if part.get("notes"):
            notes.append(str(part["notes"]))
    merged["notes"] = " | ".join(notes) if notes else None
    return merged


# ---------------------------------------------------------------- MAP

MAP_PROMPT = """You are extracting structured facts from one section of a longer document.
This is chunk {chunk_id} of {total_chunks}. You cannot see the rest of the document.

Extract ONLY what is explicitly stated in this text. Do not infer, guess, or
fill in from general knowledge. If a number, date, or claim seems cut off or
incomplete at the very start or end of the chunk, note it in "notes" instead
of guessing the missing part.

Return valid JSON, no prose, matching exactly this schema:

{{
  "entities": [
    {{"name": "string", "type": "string",
      "facts": [{{"attribute": "string", "value": "string", "unit": "string or null"}}]}}
  ],
  "superlatives": [
    {{"claim": "string", "subject": "string", "value": "string"}}
  ],
  "topics_mentioned": ["string"],
  "notes": "string or null"
}}

TEXT:
{chunk_text}
"""


def map_chunk(chunk_text: str, chunk_id: int, total_chunks: int,
              _depth: int = 0) -> dict:
    """
    Extract structured facts from a single chunk.

    Never raises: if the cheap model can't produce usable JSON even after the
    repair attempts, the chunk is halved and each half mapped separately (a
    shorter chunk means a shorter, more reliable answer). Only if that also
    fails do we return an empty extraction with a note, so one bad chunk
    can't throw away a whole ingest run.
    """
    label = f"map chunk {chunk_id}/{total_chunks}"
    if _depth:
        label += f" (split {_depth})"

    prompt = MAP_PROMPT.format(
        chunk_id=chunk_id, total_chunks=total_chunks, chunk_text=chunk_text
    )
    try:
        return call_llm_json(
            prompt, MODEL_MAP, repair_model=MODEL_REFINE, label=label
        )
    except LLMJSONError as e:
        words = chunk_text.split()
        if _depth < 2 and len(words) >= 200:
            print(f"  -> {label} failed; splitting into halves and retrying")
            mid = len(words) // 2
            return merge_extractions([
                map_chunk(" ".join(words[:mid]), chunk_id, total_chunks, _depth + 1),
                map_chunk(" ".join(words[mid:]), chunk_id, total_chunks, _depth + 1),
            ])

        path = _dump_failure(f"map_chunk_{chunk_id}_depth{_depth}", e.raw)
        print(f"  -> {label} unrecoverable; skipping. Raw output saved to {path}")
        return empty_extraction(
            note=f"Extraction failed for part of chunk {chunk_id}: {e}"
        )


# ------------------------------------------------------------- REFINE

REFINE_PROMPT = """You maintain a running fact store for a document being processed chunk by chunk.
Merge the new chunk's extracted facts into the running store.

Rules:
- Match entities by name (case-insensitive, allow minor spelling variants).
- If an entity already exists, merge new facts into its "facts" list. Do not
  duplicate the same attribute; if a NEW value conflicts with an existing one,
  add both to a "conflicts" list on that entity instead of overwriting.
- Union "superlatives" and "topics_mentioned", deduplicating exact repeats.
- Never delete or summarize away a fact.

RUNNING STORE (JSON):
{running_store}

NEW CHUNK EXTRACTION (JSON):
{chunk_extraction}

Return the updated running store as valid JSON, no prose.
"""


def refine_store(running_store: dict, chunk_extraction: dict,
                 label: str = "refine") -> dict:
    """
    Merge one chunk's facts into the running store.

    If the model's merged store won't parse, fall back to appending the new
    extraction locally: duplicated entities are much cheaper than lost facts,
    and the ANSWER prompt already scans every entity.
    """
    prompt = REFINE_PROMPT.format(
        running_store=json.dumps(running_store),
        chunk_extraction=json.dumps(chunk_extraction),
    )
    try:
        return call_llm_json(prompt, MODEL_REFINE, label=label)
    except LLMJSONError as e:
        path = _dump_failure(label.replace("/", "_").replace(" ", "_"), e.raw)
        print(f"  -> {label} failed; merging locally instead. Raw output saved to {path}")
        return merge_extractions([running_store, chunk_extraction])


# ------------------------------------------------------------- ANSWER

ANSWER_PROMPT = """Answer the question using ONLY the fact store below, compiled from
every chunk of the source document. Do not use outside knowledge.

- Aggregation/counting: check every matching entity, not just the first few.
- Superlatives: compare the relevant attribute across ALL entities that have it.
- Absence: check "topics_mentioned"; if a topic never appears, say so explicitly.
- Contradictions: look at each entity's "conflicts" field.
- Commit to a single confident answer. Do not hedge or list alternatives
  ("it might be 5, 6, or 7") -- a wrong committed answer scores better than
  a hedge, and hedging can itself trigger a forbidden claim.
- If the store lacks enough information, still give your best single answer
  rather than refusing -- an unanswered question scores zero either way.

FACT STORE (JSON):
{fact_store}

QUESTION:
{question}

Answer in 1-4 sentences, plain text, no JSON, no preamble.
"""


def answer_question(fact_store: dict, question: str) -> str:
    """Answer one question against the final fact store."""
    prompt = ANSWER_PROMPT.format(fact_store=json.dumps(fact_store), question=question)
    return call_llm(prompt, model=MODEL_ANSWER).strip()


# ------------------------------------------------------------ ORCHESTRATION

def ingest_document(text: str, words_per_chunk: int = 1500,
                     checkpoint_path: str = "fact_store.json") -> dict:
    """
    Run MAP over every chunk, then sequentially REFINE them into one
    fact store. Run this ONCE per document, then reuse the store for
    as many questions as you want. Saves a checkpoint after every chunk
    so a crash mid-ingest doesn't lose earlier work.
    """
    chunks = chunk_document(text, words_per_chunk)
    total = len(chunks)
    print(f"Split document into {total} chunks.")

    store = {"entities": [], "superlatives": [], "topics_mentioned": []}
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}/{total}] mapping...")
        extraction = map_chunk(chunk, i, total)
        print(f"[{i}/{total}] refining into store...")
        store = refine_store(store, extraction, label=f"refine chunk {i}/{total}")

        # checkpoint after every chunk -- cheap insurance against crashes
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)

    return store


def load_questions(yaml_path: str) -> list[dict]:
    """Load the practice/competition question set (id, question, ...)."""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_submission(fact_store: dict, questions: list[dict], output_path: str,
                    team: str = "yourteam", notes: str = "") -> None:
    """
    Answer every question and write the submission JSON incrementally --
    per the rules, save to disk as you go, not only at the end, so a
    partial run is still submittable.
    """
    submission = {"team": team, "notes": notes, "answers": []}

    for q in questions:
        qid = q["id"]
        print(f"Answering {qid}...")
        entry = {"id": qid}
        try:
            entry["answer"] = answer_question(fact_store, q["question"])
        except Exception as e:
            entry["answer"] = ""
            entry["error"] = str(e)
            print(f"  -> failed: {e}")

        submission["answers"].append(entry)

        # write after every single answer, not just at the end
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Submission written to {output_path} ({len(submission['answers'])} answers)")


if __name__ == "__main__":
    # 1) Ingest the document once (checkpointed as it goes)
    with open("nationalparks_europe.txt", "r", encoding="utf-8") as f:
        document_text = preprocess_document(f.read())

    fact_store = ingest_document(document_text, checkpoint_path="fact_store.json")

    # 2) Load the question set and answer every question, saving incrementally
    questions = load_questions("dev_questions.yaml")
    run_submission(
        fact_store,
        questions,
        output_path="submission.json",
        team="yourteam",
        notes="Hybrid map (ministral-3b-2512) + refine (mistral-small-3-2) fact store, then per-question answer synthesis.",
    )
