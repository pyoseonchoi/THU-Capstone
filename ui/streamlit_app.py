"""Streamlit demo interface for FULLSCAN-QA."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.question_io import parse_questions_json, parse_questions_text  # noqa: E402

st.set_page_config(page_title="FULLSCAN-QA", layout="wide")
st.title("FULLSCAN-QA")
st.caption("Operator-aware exhaustive PDF/TXT QA. Every chunk is processed.")

st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
pipeline_mode = st.sidebar.selectbox(
    "Pipeline Mode",
    ["FULLSCAN_OPERATOR", "DIRECT_CONTEXT", "SUMMARY_MAP_REDUCE", "REFINE"],
)
if st.sidebar.button("Refresh Broker Usage"):
    try:
        usage_response = httpx.get(f"{api_url}/usage", timeout=10)
        usage_response.raise_for_status()
        st.session_state["broker_usage"] = usage_response.json()
    except Exception as exc:
        st.sidebar.error(f"Usage unavailable: {exc}")

broker_usage = st.session_state.get("broker_usage", {})
if broker_usage:
    total_usage = broker_usage.get("total", {})
    st.sidebar.metric("Broker Requests", total_usage.get("requests", 0))
    st.sidebar.metric(
        "Broker Tokens",
        total_usage.get("input_tokens", 0) + total_usage.get("output_tokens", 0),
    )
    st.sidebar.metric("Broker Cost", total_usage.get("cost", 0))

tab_upload, tab_query, tab_results = st.tabs(["Upload", "Query", "Results"])

with tab_upload:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
    )
    if uploaded_file is not None and st.button("Parse Document"):
        with st.spinner("Uploading and parsing..."):
            try:
                response = httpx.post(
                    f"{api_url}/documents",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=600,
                )
                response.raise_for_status()
                data = response.json()
                st.session_state["document_id"] = data["document_id"]
                st.success("Document parsed successfully.")
                col1, col2 = st.columns(2)
                col1.metric("Pages / Segments", data["page_count"])
                col2.metric("Chunks", data["chunk_count"])
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    if "document_id" in st.session_state:
        st.info(f"Current document: `{st.session_state['document_id']}`")

with tab_query:
    st.subheader("Ask Questions")
    if "document_id" not in st.session_state:
        st.warning("Upload a document first.")
    else:
        questions_text = st.text_area(
            "Enter questions, one per line:",
            height=150,
            placeholder=(
                "How many parks have a highest point over 3000m?\n"
                "Which park has the largest area?"
            ),
        )
        question_file = st.file_uploader(
            "Question set JSON",
            type=["json"],
            key="question_set_json",
        )
        if st.button("Process Questions"):
            try:
                parsed_questions = (
                    parse_questions_json(question_file.getvalue())
                    if question_file is not None
                    else parse_questions_text(questions_text)
                )
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Invalid question set: {exc}")
                parsed_questions = []

            if not parsed_questions:
                st.warning("Enter at least one question.")
            else:
                questions = [
                    question.model_dump(mode="json")
                    for question in parsed_questions
                ]
                started = time.time()
                try:
                    llm_response = httpx.get(
                        f"{api_url}/health/llm",
                        timeout=5,
                    )
                    llm_response.raise_for_status()
                    llm_status = llm_response.json()
                    if not llm_status.get("available", False):
                        raise RuntimeError(llm_status.get(
                            "message",
                            "The configured LLM is unavailable.",
                        ))
                    response = httpx.post(
                        f"{api_url}/answer-jobs",
                        json={
                            "document_id": st.session_state["document_id"],
                            "questions": questions,
                            "pipeline_mode": pipeline_mode,
                            "include_diagnostics": True,
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    job_id = response.json()["job_id"]
                    progress = st.progress(0.0, text="Planning questions...")

                    while True:
                        elapsed = time.time() - started
                        if elapsed > 3600:
                            raise TimeoutError(
                                "The job exceeded the 60 minute UI limit."
                            )
                        status_response = httpx.get(
                            f"{api_url}/answer-jobs/{job_id}", timeout=30
                        )
                        status_response.raise_for_status()
                        job = status_response.json()
                        total = job.get("total", 0)
                        processed = job.get("processed", 0)
                        ratio = processed / total if total else 0.0
                        progress.progress(
                            min(1.0, ratio),
                            text=(
                                f"Mapped {processed}/{total} batches; "
                                f"failures queued: {job.get('failed', 0)}"
                            ),
                        )
                        if job["status"] == "completed":
                            st.session_state["last_result"] = job["result"]
                            st.session_state["last_elapsed"] = elapsed
                            progress.progress(1.0, text="Completed")
                            st.success(f"Completed in {elapsed:.1f}s")
                            break
                        if job["status"] == "failed":
                            raise RuntimeError(job.get("error", "Job failed"))
                        time.sleep(2)
                except Exception as exc:
                    st.error(f"Question processing failed: {exc}")

with tab_results:
    st.subheader("Results")
    if "last_result" not in st.session_state:
        st.info("No results yet.")
    else:
        result = st.session_state["last_result"]
        elapsed = st.session_state.get("last_elapsed", 0)
        usage = result.get("usage", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Questions", len(result.get("answers", [])))
        col2.metric("Input Tokens", usage.get("total_input_tokens", "N/A"))
        col3.metric("Output Tokens", usage.get("total_output_tokens", "N/A"))
        col4.metric("Elapsed", f"{elapsed:.1f}s")

        st.divider()
        for answer in result.get("answers", []):
            label = f"{answer['question_id']} - {answer.get('operator', 'N/A')}"
            with st.expander(label, expanded=True):
                st.markdown(f"**Answer:** {answer.get('final_answer', 'N/A')}")
                if answer.get("answer_with_evidence"):
                    st.markdown(
                        f"**With evidence:** {answer['answer_with_evidence']}"
                    )
                coverage = answer.get("coverage", {})
                if coverage:
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Coverage",
                        f"{coverage.get('coverage_percentage', 'N/A')}%",
                    )
                    c2.metric(
                        "Entities", coverage.get("detected_entity_count", "N/A")
                    )
                    c3.metric("Failed Chunks", coverage.get("failed_chunks", 0))
                    for warning in coverage.get("warnings", []):
                        st.warning(warning)
                for warning in answer.get("warnings", []):
                    st.warning(warning)

        st.divider()
        submission = result.get("submission", {})
        st.download_button(
            "Download Submission JSON",
            json.dumps(submission, indent=2, ensure_ascii=False),
            file_name="submission.json",
            mime="application/json",
            type="primary",
        )
        st.download_button(
            "Download Full Diagnostics (JSON)",
            json.dumps(result, indent=2, default=str),
            file_name="fullscan_diagnostics.json",
            mime="application/json",
        )
