# LogiMind

Multi-agent RAG system for querying DHL's public operational documents in natural language — rate guides, customs guidelines, packing rules, prohibited items, incoterms, annual reports, sustainability report, and strategy documents.

Live demo: https://ui-production-b575.up.railway.app (rate-limited to 10 requests/minute)

## What it does

Ask something like "what items are prohibited from shipping with DHL?" or "where is my package with tracking number X?" and get a cited answer pulled from the actual source documents, or a simulated shipment status. Out-of-scope questions get refused rather than answered from general knowledge — the system only answers from what it actually retrieved.

## Architecture

### Ingestion (one-time batch)

```mermaid
flowchart LR
    A[sources.py] --> B[downloader.py]
    B --> C[loader.py - PyMuPDF]
    C --> D[chunker.py]
    D --> E[embedder.py - OpenAI]
    E --> F[store.py]
    F --> G[(Qdrant)]
    F --> H[(SQLite)]
```

- 14 PDFs, 874 pages, 5,133 chunks (512 chars, 64 overlap)
- Loader strips repeated per-page boilerplate (headers/footers) by detecting lines that repeat across most pages, rather than hardcoding patterns per document
- Every page was checked for extractable text before deciding whether OCR support was needed — all 874 had usable text, so OCR handling was left out instead of built speculatively
- Downloader retries with exponential backoff and skips already-downloaded files
- Embedding calls batched at 100 chunks/request; Qdrant writes batched separately at 200 points/request after hitting its 32MB per-request limit at full scale

### Query pipeline (runtime)

```mermaid
flowchart LR
    U[User] --> UI[Streamlit]
    UI --> API[FastAPI /query]
    API --> O[Orchestrator]
    O --> P[PlannerAgent - Claude]
    P --> R[RetrieverAgent - deterministic]
    R -->|knowledge search| H[hybrid.py]
    H --> G[(Qdrant)]
    H --> S[(SQLite / BM25)]
    R -->|tracking lookup| T[tracking tool]
    R --> Resp[ResponseAgent - Claude]
    Resp --> API
```

- Hybrid retrieval: Qdrant vector search and BM25 keyword search run independently, get deduplicated, then re-ranked with a cross-encoder (`ms-marco-MiniLM-L-6-v2`). The two methods surface genuinely different useful chunks on the same query, which is the actual point of running both instead of just one.
- PlannerAgent and ResponseAgent are Claude-backed AutoGen agents. RetrieverAgent deliberately isn't — it's plain deterministic Python, since Planner's decision already fully determines what needs to run. The orchestrator calls all three in a fixed sequence rather than using AutoGen's group-chat abstraction, since there's no open-ended agent-to-agent conversation to manage.
- Two tools available to the retrieval step: a knowledge search wrapping the hybrid retriever, and a simulated tracking lookup that deterministically derives a status from the tracking number so results stay consistent across repeat calls.
- LangSmith traces every step of the pipeline.
- `/query` is rate-limited (10/minute per client IP, proxy-aware) and caps question length, since every call spends real OpenAI/Anthropic budget.

### Deployment

Two separate Docker images rather than one: the API image carries the full ML stack (CPU-only torch build, not the default CUDA one); the UI image only needs `httpx` and `streamlit`, since it's just an HTTP client to the API. Deployed as two services on Railway.

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
