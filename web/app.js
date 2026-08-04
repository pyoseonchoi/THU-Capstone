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
}

document.addEventListener("DOMContentLoaded", initHandlers);
