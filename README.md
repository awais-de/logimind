# LogiMind

Multi-agent RAG system for querying DHL's public operational documents in natural language — rate guides, customs guidelines, packing rules, prohibited items, incoterms, annual reports, sustainability report, and strategy documents.

Live demo: https://ui-production-b575.up.railway.app (rate-limited to 10 requests/minute)

## What it does

Ask something like "what items are prohibited from shipping with DHL?" or "where is my package with tracking number X?" and get a cited answer pulled from the actual source documents, or a simulated shipment status. Out-of-scope questions get refused rather than answered from general knowledge.

## Pipeline

**Ingestion** (`data/`): 14 DHL PDFs → PyMuPDF text extraction → chunked (512 chars, 64 overlap) → embedded with OpenAI `text-embedding-3-small` → stored in Qdrant (vectors) and SQLite (metadata index). 5,133 chunks total.

**Retrieval** (`retrieval/`): hybrid search — Qdrant vector search plus BM25 keyword search, merged and re-ranked with a sentence-transformers cross-encoder.

**Agents** (`agents/`): three roles, not a free-form AutoGen group chat. A Claude-backed planner decides what's needed (knowledge search, tracking lookup, both, neither); a plain deterministic retriever executes that decision by calling the actual tools; a Claude-backed responder synthesizes the final answer with citations. Full trace capture via LangSmith.

**API/UI**: FastAPI backend (`api/`), Streamlit frontend (`ui/`), deployed as two separate Docker images — the UI image skips the ML dependencies entirely since it's just an HTTP client to the API.

## Stack

Python 3.12, Pydantic, PyMuPDF, OpenAI (embeddings), Anthropic Claude + AutoGen (agents), Qdrant, BM25s, sentence-transformers, FastAPI, Streamlit, LangSmith, pytest.

## Running it locally

```
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
python -m data.ingestion.store   # one-time: download, chunk, embed, store
uvicorn api.main:app --reload
streamlit run ui/app.py
```

The UI only needs `requirements-ui.txt`.

Or with Docker: `docker compose up --build`.

## Testing

`pytest` — 90 tests, all external calls mocked so the suite runs without hitting real APIs.

## Status

Ingestion, retrieval, and the agent pipeline are done and deployed. Still open: a RAGAS evaluation harness for retrieval/answer quality, latency/cost tracking, and CI.
