// web/app.js — talks to the FastAPI backend directly; no build step, no framework.

let API_BASE = "http://127.0.0.1:8003";

const state = {
  documentId: null,
  documentFilename: "",
  displayDocumentId: null, // optional: a separate PDF used only for the docked viewer
  chunkCount: 0,
  questions: [], // [{question_id, question, category}]
  selectedQuestionId: null,
  submission: null,
  answersById: new Map(), // question_id -> full answer payload (evidence, coverage, ...)
  modelUsageTotals: new Map(), // model -> {input_tokens, output_tokens}
};

function el(id) {
  return document.getElementById(id);
}

function generateQuestionId() {
  return `q-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

if (typeof pdfjsLib !== "undefined") {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
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

function addFileRow(ext, name, size, stateLabel, chipClass, onRemove) {
  const row = document.createElement("div");
  row.className = "file";
  row.innerHTML = `
    <div class="ext ${ext.toLowerCase() === "txt" ? "txt" : "yaml"}">${ext}</div>
    <div class="name">${name}</div>
    <div class="size mono">${size}</div>
    <div class="state"><span class="chip ${chipClass}">${stateLabel}</span></div>
    <button class="file-remove" type="button" aria-label="Remove ${name}">&times;</button>
  `;
  row.querySelector(".file-remove").addEventListener("click", () => {
    row.remove();
    if (onRemove) onRemove();
  });
  el("file-list").appendChild(row);
}

function maybeEnableRunButtons() {
  const hasDocument = Boolean(state.documentId);
  const hasQuestions = state.questions.length > 0;
  const hasSelection = Boolean(getSelectedQuestion());
  el("run-selected-btn").disabled = !(hasDocument && hasSelection);
  el("run-all-btn").disabled = !(hasDocument && hasQuestions);
  el("freeform-ask-btn").disabled = !hasDocument;
  el("freeform-input").disabled = !hasDocument;
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
      throw new Error(detail.detail || `Upload failed (${res.status})`);
    }
    const doc = await res.json();
    state.documentId = doc.document_id;
    state.documentFilename = doc.filename;
    state.chunkCount = doc.chunk_count;
    renderDocumentUploaded(doc, file);
  } catch (err) {
    showError(err.message);
  }
}

async function uploadDisplayPdf(file) {
  clearError();
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch(`${API_BASE}/documents`, { method: "POST", body: formData });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Upload failed (${res.status})`);
    }
    const doc = await res.json();
    // Answers still come from state.documentId (e.g. the .txt) -- this PDF is
    // only ever fetched by the docked viewer, so invalidate its cache to
    // force a re-fetch under the new id next time something is viewed.
    state.displayDocumentId = doc.document_id;
    sourceViewer.documentId = null;
    sourceViewer.kind = null;
    el("viewer-sub").textContent = "Preview";
    openSourceViewer("", null);
  } catch (err) {
    showError(err.message);
  }
}

function renderDocumentUploaded(doc, file) {
  addFileRow(
    file.name.split(".").pop().toUpperCase(),
    file.name,
    formatBytes(file.size),
    "uploaded",
    "chip-ok",
    removeDocument
  );
  el("chunk-summary").style.display = "flex";
  el("chunk-summary-count").textContent = doc.chunk_count;
  el("stat-chunks").textContent = doc.chunk_count;
  el("stat-chunks-status").textContent = "uploaded";
  el("index-doc-label").textContent = `${doc.filename} · ${doc.chunk_count} chunks`;
  el("index-bar").style.width = "100%";
  el("index-chunk-count").innerHTML = `<b>${doc.chunk_count}</b> / <b>${doc.chunk_count}</b>`;
  const statusValue = el("status-toggle-value");
  if (statusValue) statusValue.textContent = `${doc.chunk_count} / ${doc.chunk_count}`;
  maybeEnableRunButtons();
  openSourceViewer("", null);
}

function removeDocument() {
  state.documentId = null;
  state.documentFilename = "";
  state.displayDocumentId = null;
  state.chunkCount = 0;
  el("chunk-summary").style.display = "none";
  el("stat-chunks").textContent = "0";
  el("stat-chunks-status").textContent = "waiting for upload";
  el("index-doc-label").textContent = "Upload a document";
  el("index-bar").style.width = "0%";
  el("index-chunk-count").innerHTML = "0 / 0";
  const statusValue = el("status-toggle-value");
  if (statusValue) statusValue.textContent = "0 / 0";
  el("doc-input").value = "";
  sourceViewer.documentId = null;
  sourceViewer.kind = null;
  el("source-pdf-wrap").style.display = "none";
  el("source-text-pane").style.display = "none";
  el("viewer-sub").textContent = "";
  const viewerEmpty = el("viewer-empty");
  viewerEmpty.style.display = "block";
  viewerEmpty.textContent = "Upload a document to preview it here.";
  maybeEnableRunButtons();
}

// ---------- question set ----------

function isYamlFile(file) {
  return /\.(ya?ml)$/i.test(file.name);
}

function parseQuestionData(text, file) {
  if (isYamlFile(file)) {
    if (typeof jsyaml === "undefined") {
      throw new Error("YAML parser failed to load (check your internet connection) — try a .json file instead.");
    }
    return jsyaml.load(text);
  }
  return JSON.parse(text);
}

async function loadQuestions(file) {
  clearError();
  try {
    const text = await file.text();
    const data = parseQuestionData(text, file);
    const items = Array.isArray(data) ? data : data.questions;
    if (!Array.isArray(items)) {
      throw new Error("The file must be a list of questions, or an object with a questions list.");
    }
    const questions = items.map((item) => ({
      question_id: String(item.id ?? item.question_id ?? ""),
      question: String(item.question ?? item.text ?? "").trim(),
      category: String(item.category ?? ""),
    }));
    if (questions.some((q) => !q.question_id || !q.question)) {
      throw new Error("Every question needs an id and question text.");
    }
    state.questions = questions;
    state.selectedQuestionId = questions[0] ? questions[0].question_id : null;
    const ext = isYamlFile(file) ? "YAML" : "JSON";
    addFileRow(ext, file.name, formatBytes(file.size), `${questions.length} questions`, "chip-cat", removeQuestions);
    el("stat-question-count").textContent = questions.length;
    populateQuestionPicker();
    maybeEnableRunButtons();
  } catch (err) {
    showError(`Could not read the question file: ${err.message}`);
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
  picker.value = state.selectedQuestionId || "";
  picker.size = 5;
  el("question-picker-summary").textContent = `${state.questions.length} questions loaded · showing 5 at a time`;
  updatePickedCategory();
}

function updatePickedCategory() {
  const picker = el("question-picker");
  if (picker && picker.value) state.selectedQuestionId = picker.value;
  const picked = getSelectedQuestion();
  el("picked-category").textContent = picked && picked.category ? picked.category : "uncategorized";
}

function removeQuestions() {
  state.questions = [];
  state.selectedQuestionId = null;
  const picker = el("question-picker");
  picker.innerHTML = "<option>Upload a question set first</option>";
  picker.disabled = true;
  picker.size = 1;
  el("question-picker-summary").textContent = "Upload a question set to browse questions.";
  el("picked-category").textContent = "uncategorized";
  el("stat-question-count").textContent = "0";
  el("questions-input").value = "";
  maybeEnableRunButtons();
}

function getSelectedQuestion() {
  return state.questions.find((q) => q.question_id === state.selectedQuestionId) || null;
}

// ---------- stage -> model badge (see spec's confirmed lookup table) ----------

const STAGE_INFO = {
  compiling: { label: "Compiling document", badge: null },
  mapping: { label: "Extracting evidence (exhaustive mapping)", badge: "3B" },
  repairing: { label: "Repairing mapping", badge: "24B" },
  answering: { label: "Synthesizing final answer", badge: "24B" },
};

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

function submitFreeformQuestion() {
  const input = el("freeform-input");
  const text = input.value.trim();
  if (!text) return;
  const question = { question_id: generateQuestionId(), question: text, category: "" };
  input.value = "";
  startRun([question]);
}

// ---------- run ----------

async function startRun(questions) {
  clearError();
  questions.forEach((q) => appendChatTurn(q.question, q.question_id));
  if (questions.length > 1) {
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

// ---------- result rendering ----------
// Every /answer-jobs call only answers the questions it was given -- it is
// NOT cumulative on the server. If we replaced state.submission wholesale on
// every run, downloading right after a single "Run this question" test would
// silently produce a submission.json containing just that one answer. So we
// merge answers by id into a running submission instead, and surface how
// many of the loaded questions are actually covered before download.

function mergeSubmission(newSubmission) {
  const merged = state.submission ? { ...state.submission } : { answers: [] };
  if (newSubmission.team) merged.team = newSubmission.team;
  if (newSubmission.notes) merged.notes = newSubmission.notes;
  const byId = new Map((merged.answers || []).map((a) => [a.id, a]));
  (newSubmission.answers || []).forEach((a) => byId.set(a.id, a));
  merged.answers = Array.from(byId.values());
  state.submission = merged;
}

function updateSubmissionProgressLabel() {
  const answered = state.submission ? state.submission.answers.length : 0;
  const total = state.questions.length;
  el("submission-progress-label").textContent = total
    ? `${answered} / ${total} answered`
    : "";
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

// ---------- evidence rendering ----------

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderEvidenceList(container, answer) {
  container.innerHTML = "";
  const quotes = answer.evidence_quotes || [];
  if (!quotes.length) {
    container.innerHTML = '<div class="evidence-empty">No source quotes were recorded for this answer.</div>';
    return;
  }
  const label = document.createElement("div");
  label.className = "evidence-list-label";
  label.textContent = `Evidence (${quotes.length})`;
  container.appendChild(label);

  quotes.forEach((item) => {
    const row = document.createElement("div");
    row.className = "evidence-quote";
    const pageKnown = item.page !== null && item.page !== undefined;
    row.innerHTML = `
      <div class="evidence-quote-text">${escapeHtml(item.quote)}</div>
      <div class="evidence-quote-actions">
        <span class="chip evidence-page-chip ${pageKnown ? "" : "unknown"}">${pageKnown ? `p.${item.page}` : "page n/a"}</span>
        <button class="view-source-btn" type="button">View in source</button>
      </div>
    `;
    row.querySelector(".view-source-btn").addEventListener("click", () => {
      openSourceViewer(item.quote, pageKnown ? item.page : null);
    });
    container.appendChild(row);
  });
}

// ---------- source viewer (PDF highlight / TXT highlight) ----------

const sourceViewer = {
  documentId: null,
  filename: "",
  kind: null, // "pdf" | "txt"
  pdfDoc: null,
  pdfPage: 1,
  pdfPageCount: 0,
  targetQuote: "",
  txtContent: "",
};

function normalizeForSearch(text) {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

// Evidence strings sometimes carry an inline "(page N)" suffix (absence-topic
// quotes) or a "Label: " prefix (fact-card quotes) that never appears in the
// source text verbatim -- strip those before searching so the match succeeds.
function coreQuoteText(quote) {
  return quote
    .replace(/\s*\(page\s+\d+\)\s*$/i, "")
    .replace(/^[^:]{1,40}:\s*/, "")
    .trim();
}

async function openSourceViewer(quote, page) {
  if (!state.documentId) return;
  const empty = el("viewer-empty");
  el("source-pdf-wrap").style.display = "none";
  el("source-text-pane").style.display = "none";
  el("viewer-sub").textContent = page ? `Page ${page}` : (quote ? "Searching…" : "Preview");

  sourceViewer.targetQuote = coreQuoteText(quote);
  const viewDocumentId = state.displayDocumentId || state.documentId;

  if (sourceViewer.documentId !== viewDocumentId || !sourceViewer.kind) {
    empty.style.display = "block";
    empty.textContent = "Loading source…";
    try {
      const res = await fetch(`${API_BASE}/documents/${viewDocumentId}/file`);
      if (!res.ok) throw new Error(`Could not load the source file (${res.status})`);
      const contentType = res.headers.get("content-type") || "";
      sourceViewer.documentId = viewDocumentId;
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

function renderTxtHighlight() {
  const pane = el("source-text-pane");
  // The machine-readable .txt export uses literal "\n" (backslash + n) as its
  // paragraph separator rather than real newline bytes -- decode it once here
  // so the raw dump reads like text instead of one run-on line of escapes.
  const full = sourceViewer.txtContent.replace(/\\n/g, "\n");
  const target = sourceViewer.targetQuote;
  const idx = target ? full.toLowerCase().indexOf(target.toLowerCase()) : -1;
  if (idx === -1) {
    pane.textContent = full;
    return;
  }
  const before = full.slice(0, idx);
  const match = full.slice(idx, idx + target.length);
  const after = full.slice(idx + target.length);
  pane.innerHTML = `${escapeHtml(before)}<mark id="txt-highlight">${escapeHtml(match)}</mark>${escapeHtml(after)}`;
  const mark = el("txt-highlight");
  if (mark) mark.scrollIntoView({ block: "center" });
}

async function renderPdfPage() {
  const page = await sourceViewer.pdfDoc.getPage(sourceViewer.pdfPage);
  // Fit the page to the docked panel's actual width instead of a fixed scale
  // -- the panel is a narrow column now, not the old wide modal, so a fixed
  // 1.4x rendered far smaller than the space available to it.
  const unscaled = page.getViewport({ scale: 1 });
  const isLightboxOpen = el("pdf-lightbox").classList.contains("open");
  const fitContainer = isLightboxOpen ? el("pdf-lightbox-body") : el("viewer-body");
  const availableWidth = fitContainer.clientWidth - 24;
  const fitScale = Math.max(availableWidth / unscaled.width, 0.5);
  const viewport = page.getViewport({ scale: fitScale });
  const canvas = el("source-pdf-canvas");
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;
  el("source-pdf-page-label").textContent = `${sourceViewer.pdfPage} / ${sourceViewer.pdfPageCount}`;

  const layer = el("source-pdf-highlight-layer");
  layer.innerHTML = "";
  layer.style.width = `${viewport.width}px`;
  layer.style.height = `${viewport.height}px`;
  if (!sourceViewer.targetQuote) return;

  const textContent = await page.getTextContent();
  const target = normalizeForSearch(sourceViewer.targetQuote);
  // Build the page's running text alongside the item that produced each
  // character, so a match found in the joined string can be traced back to
  // the specific text-layer items whose boxes need highlighting.
  let running = "";
  const spans = [];
  textContent.items.forEach((item) => {
    const start = running.length;
    running += normalizeForSearch(item.str) + " ";
    spans.push({ item, start, end: running.length });
  });
  const matchStart = target.length > 3 ? running.indexOf(target.slice(0, Math.min(target.length, 120))) : -1;
  if (matchStart === -1) return;
  const matchEnd = matchStart + Math.min(target.length, 120);

  // Each text-layer item becomes its own tiny rect; drawing one bordered box
  // per item looks like a broken staircase. Convert to rects first, then
  // merge rects that sit on the same visual line into one continuous bar.
  const rects = spans
    .filter((span) => span.end > matchStart && span.start < matchEnd)
    .map((span) => {
      const tx = pdfjsLib.Util.transform(viewport.transform, span.item.transform);
      const fontHeight = Math.hypot(tx[2], tx[3]);
      return {
        left: tx[4],
        top: tx[5] - fontHeight,
        right: tx[4] + span.item.width * viewport.scale,
        bottom: tx[5] - fontHeight + fontHeight * 1.2,
      };
    })
    .sort((a, b) => a.top - b.top || a.left - b.left);

  const lines = [];
  rects.forEach((rect) => {
    const line = lines.find((candidate) => Math.abs(candidate.top - rect.top) < 4);
    if (line) {
      line.left = Math.min(line.left, rect.left);
      line.top = Math.min(line.top, rect.top);
      line.right = Math.max(line.right, rect.right);
      line.bottom = Math.max(line.bottom, rect.bottom);
    } else {
      lines.push({ ...rect });
    }
  });

  lines.forEach((line) => {
    const highlight = document.createElement("div");
    highlight.className = "source-pdf-highlight";
    highlight.style.left = `${line.left - 2}px`;
    highlight.style.top = `${line.top}px`;
    highlight.style.width = `${line.right - line.left + 4}px`;
    highlight.style.height = `${line.bottom - line.top}px`;
    layer.appendChild(highlight);
  });
  const firstHighlight = layer.querySelector(".source-pdf-highlight");
  if (firstHighlight) firstHighlight.scrollIntoView({ block: "center" });
}

function openPdfLightbox() {
  if (sourceViewer.kind !== "pdf") return;
  el("pdf-lightbox-body").appendChild(el("source-pdf-canvas-wrap"));
  el("pdf-lightbox").classList.add("open");
  el("pdf-lightbox").setAttribute("aria-hidden", "false");
  renderPdfPage();
}

function closePdfLightbox() {
  el("pdf-lightbox").classList.remove("open");
  el("pdf-lightbox").setAttribute("aria-hidden", "true");
  el("source-pdf-wrap").appendChild(el("source-pdf-canvas-wrap"));
  if (sourceViewer.kind === "pdf") renderPdfPage();
}

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
  el("source-pdf-expand").addEventListener("click", openPdfLightbox);
  el("pdf-lightbox-close").addEventListener("click", closePdfLightbox);
  el("pdf-lightbox-backdrop").addEventListener("click", closePdfLightbox);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && el("pdf-lightbox").classList.contains("open")) closePdfLightbox();
  });
}

function downloadSubmission() {
  if (!state.submission) return;
  const answered = state.submission.answers.length;
  const total = state.questions.length;
  if (total && answered < total) {
    const proceed = confirm(
      `Only ${answered} of ${total} loaded questions have been answered so far. ` +
      `Download the partial submission.json anyway?`
    );
    if (!proceed) return;
  }
  const blob = new Blob([JSON.stringify(state.submission, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "submission.json";
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- usage polling ----------
// Confirmed live against the real broker (2026-08-04):
//   {"group": "...", "models": {"ministral-3b-2512": {"requests":.., "input_tokens":..,
//   "output_tokens":.., "cost":..}, "mistral-small-3-2": {...}}, "total": {..., "cost":..},
//   "legacy": {}}
// "total.cost" is the group's cumulative spend and already includes both models' cost --
// use it directly rather than summing per-model costs (which would double-count).

function formatUsageCost(usage) {
  if (usage.total && typeof usage.total.cost === "number") {
    return `$${usage.total.cost.toFixed(2)}`;
  }
  // Fallbacks in case the broker's response shape ever changes.
  if (typeof usage.total_cost === "number") return `$${usage.total_cost.toFixed(2)}`;
  if (typeof usage.cost === "number") return `$${usage.cost.toFixed(2)}`;
  const perModel = usage.models || usage.by_model;
  if (perModel && typeof perModel === "object") {
    const sum = Object.values(perModel).reduce(
      (s, m) => s + (m && typeof m.cost === "number" ? m.cost : 0),
      0
    );
    if (sum > 0) return `$${sum.toFixed(2)}`;
  }
  return "n/a";
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

// ---------- pipeline parameters (read-only) ----------

const SETTINGS_LABELS = {
  mapper_model: "Mapper model (3B)",
  answer_model: "Answer/repair model (24B)",
  question_batch_size: "Question batch size",
  record_batch_size: "Record batch size",
  max_concurrent_requests: "Max concurrent requests",
  request_timeout_seconds: "Request timeout (s)",
  max_retries: "Max retries",
  pipeline_mode: "Pipeline mode",
  team_name: "Team name",
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

// ---------- wiring ----------

function initStatusWidget() {
  const widget = document.querySelector(".status-widget");
  const toggle = el("status-toggle");
  const panel = el("status-panel");
  if (!widget || !toggle) return;

  const setOpen = (open) => {
    widget.classList.toggle("status-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    if (panel) panel.setAttribute("aria-hidden", String(!open));
  };

  toggle.addEventListener("click", () => {
    setOpen(!widget.classList.contains("status-open"));
  });

  document.addEventListener("click", (event) => {
    if (!widget.contains(event.target)) setOpen(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setOpen(false);
  });
}

// ---------- model backend switch ----------

function switchBackend(newBase) {
  if (newBase === API_BASE) return;
  API_BASE = newBase;

  // A different backend is a different process with its own document/run
  // store -- nothing from the old session (document, chat thread, evidence,
  // submission, usage totals) is valid against it, so clear everything
  // rather than leave stale state pointing at IDs the new server has never
  // seen.
  removeDocument();
  removeQuestions();
  state.submission = null;
  state.answersById.clear();
  state.modelUsageTotals.clear();
  renderPerformanceStats();
  el("chat-thread").innerHTML = '<div class="chat-empty" id="chat-empty">Pick a question above, or type your own, to get started.</div>';
  el("download-submission-btn").disabled = true;
  el("submission-progress-label").textContent = "";
  el("stat-tokens").textContent = "—";
  el("stat-progress").textContent = "0/0";
  el("file-list").innerHTML = "";

  pollUsage();
  loadPipelineSettings();
}

function initHandlers() {
  el("backend-select").addEventListener("change", (e) => {
    switchBackend(e.target.value);
  });
  el("doc-input").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadDocument(e.target.files[0]);
  });
  el("display-pdf-input").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadDisplayPdf(e.target.files[0]);
  });
  el("doc-dropzone").addEventListener("dragover", (e) => e.preventDefault());
  el("doc-dropzone").addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) uploadDocument(e.dataTransfer.files[0]);
  });

  el("questions-input").addEventListener("change", (e) => {
    if (e.target.files[0]) loadQuestions(e.target.files[0]);
  });
  el("questions-dropzone").addEventListener("dragover", (e) => e.preventDefault());
  el("questions-dropzone").addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) loadQuestions(e.dataTransfer.files[0]);
  });

  el("question-picker").addEventListener("change", () => {
    updatePickedCategory();
    maybeEnableRunButtons();
  });

  el("run-selected-btn").addEventListener("click", () => {
    const question = getSelectedQuestion();
    if (question) startRun([question]);
  });

  el("run-all-btn").addEventListener("click", () => {
    startRun(state.questions);
  });

  el("freeform-ask-btn").addEventListener("click", submitFreeformQuestion);
  el("freeform-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !el("freeform-ask-btn").disabled) submitFreeformQuestion();
  });

  el("download-submission-btn").addEventListener("click", downloadSubmission);
}

document.addEventListener("DOMContentLoaded", () => {
  initStatusWidget();
  initSourceViewer();
  initHandlers();
  pollUsage();
  setInterval(pollUsage, 5000);
  loadPipelineSettings();
});
