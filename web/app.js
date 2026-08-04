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
      throw new Error(detail.detail || `Upload failed (${res.status})`);
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
  addFileRow(file.name.split(".").pop().toUpperCase(), file.name, formatBytes(file.size), "uploaded", "chip-ok");
  el("chunk-summary").style.display = "flex";
  el("chunk-summary-count").textContent = doc.chunk_count;
  el("stat-chunks").textContent = doc.chunk_count;
  el("stat-chunks-status").textContent = "uploaded";
  el("index-doc-label").textContent = `${doc.filename} · ${doc.chunk_count} chunks`;
  el("index-bar").style.width = "100%";
  el("index-chunk-count").innerHTML = `<b>${doc.chunk_count}</b> / <b>${doc.chunk_count}</b>`;
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
    const ext = isYamlFile(file) ? "YAML" : "JSON";
    addFileRow(ext, file.name, formatBytes(file.size), `${questions.length} questions`, "chip-cat");
    el("questions-count-label").textContent = `${questions.length} questions loaded`;
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
  updatePickedCategory();
}

function updatePickedCategory() {
  const picker = el("question-picker");
  const picked = state.questions.find((q) => q.question_id === picker.value);
  el("picked-category").textContent = picked && picked.category ? picked.category : "uncategorized";
}

// ---------- stage -> model badge (see spec's confirmed lookup table) ----------

const STAGE_INFO = {
  compiling: { label: "Compiling document", badge: null },
  mapping: { label: "Extracting evidence (exhaustive mapping)", badge: "3B" },
  repairing: { label: "Repairing mapping", badge: "24B" },
  answering: { label: "Synthesizing final answer", badge: "24B" },
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
      throw new Error(detail.detail || `Failed to start run (${res.status})`);
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
        showError(job.error || "The run failed.");
      }
    } catch (err) {
      showError(`Failed to load results: ${err.message}`);
    }
  });

  source.onerror = () => {
    showError("Progress log connection lost. Try reloading to reconnect — progress is preserved on the server.");
  };
}

function appendLogRow(data) {
  const info = STAGE_INFO[data.stage] || { label: data.stage, badge: null };
  const li = document.createElement("li");
  li.dataset.badge = info.badge || "";
  const badgeHtml = info.badge
    ? `<span class="chip ${info.badge === "3B" ? "chip-3b" : "chip-24b"}">${info.badge}</span>`
    : "";
  const failedLabel = data.failed ? `, ${data.failed} failed` : "";
  li.innerHTML = `
    <div class="dot active"></div>
    <div>
      <div class="step">${info.label} (${data.processed}/${data.total}${failedLabel})</div>
      <div class="meta">${badgeHtml}<span class="time mono">${new Date().toLocaleTimeString("en-US")}</span></div>
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
      <td><span class="chip chip-cat">${question && question.category ? question.category : "uncategorized"}</span></td>
      <td><span class="status-pill ${failed ? "" : "ok"}">${failed ? "failed" : "done"}</span></td>
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
    <span class="chip chip-cat">${question && question.category ? question.category : "uncategorized"}</span>
    ${failed ? `<span class="chip chip-err">${answer.warnings[0]}</span>` : `<span class="chip chip-ok">done</span>`}
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

// ---------- pipeline parameters (read-only) ----------

const SETTINGS_LABELS = {
  mapper_model: "Mapper model (3B)",
  answer_model: "Answer/repair model (24B)",
  verifier_model: "Verifier model",
  planner_model: "Planner model",
  chunk_target_tokens: "Chunk target tokens",
  chunk_overlap_tokens: "Chunk overlap tokens",
  question_batch_size: "Question batch size",
  record_batch_size: "Record batch size",
  max_concurrent_requests: "Max concurrent requests",
  request_timeout_seconds: "Request timeout (s)",
  max_retries: "Max retries",
  pipeline_mode: "Pipeline mode",
  llm_temperature: "temperature",
  llm_top_p: "top_p",
  llm_seed: "seed",
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

function initHandlers() {
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

  el("download-submission-btn").addEventListener("click", downloadSubmission);
}

document.addEventListener("DOMContentLoaded", () => {
  initHandlers();
  pollUsage();
  setInterval(pollUsage, 5000);
  loadPipelineSettings();
});
