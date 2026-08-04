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
});
