"""Streamlit chat frontend for LogiMind."""

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("LOGIMIND_API_URL", "http://localhost:8000")


def _api_status() -> bool:
    """Check whether the FastAPI backend is reachable."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _render_trace(data: dict) -> None:
    """Render the agent trace summary, tracking details, and citations."""
    actions = []
    if data.get("needs_knowledge_search"):
        actions.append("Searched DHL's knowledge base")
    if data.get("needs_tracking_lookup"):
        actions.append("Looked up shipment tracking")
    if actions:
        st.caption(" · ".join(actions))

    tracking_info = data.get("tracking_info")
    if tracking_info:
        with st.expander("Tracking details"):
            st.json(tracking_info)

    citations = data.get("citations") or []
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            for citation in citations:
                st.markdown(
                    f"**{citation['doc_name']}**, p.{citation['page_number']} "
                    f"— score {citation['score']:.2f}"
                )
                st.caption(citation["text_snippet"])


def _submit_feedback(query_id: str, vote: str) -> bool:
    """POST a thumbs up/down vote to the API. Returns True on success."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/feedback", json={"query_id": query_id, "vote": vote}, timeout=5.0
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _render_feedback(query_id: str | None) -> None:
    """Render thumbs up/down controls for one answer, once per query_id."""
    if not query_id:
        return

    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}

    given = st.session_state.feedback_given.get(query_id)
    if given:
        st.caption(f"Thanks for the feedback ({'👍' if given == 'up' else '👎'})")
        return

    up_col, down_col, _ = st.columns([1, 1, 8])
    if up_col.button("👍", key=f"feedback-up-{query_id}"):
        if _submit_feedback(query_id, "up"):
            st.session_state.feedback_given[query_id] = "up"
            st.rerun()
    if down_col.button("👎", key=f"feedback-down-{query_id}"):
        if _submit_feedback(query_id, "down"):
            st.session_state.feedback_given[query_id] = "down"
            st.rerun()


st.set_page_config(page_title="LogiMind", page_icon=":package:")

with st.sidebar:
    st.title("LogiMind")
    st.caption(
        "Ask questions about DHL's public operational documents, or check a "
        "(simulated) shipment's tracking status."
    )
    st.divider()
    if _api_status():
        st.success("API connected")
    else:
        st.error(f"API unreachable at {API_BASE_URL}")

st.title("LogiMind")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            _render_trace(message["trace"])
            _render_feedback(message["trace"].get("query_id"))

question = st.chat_input("Ask a question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        data = None
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/query", json={"question": question}, timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")

        if data is not None:
            st.markdown(data["answer"])
            _render_trace(data)
            _render_feedback(data.get("query_id"))
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "trace": data}
            )
