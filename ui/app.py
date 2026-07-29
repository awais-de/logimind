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
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "trace": data}
            )
