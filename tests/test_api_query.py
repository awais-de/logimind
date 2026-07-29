"""Unit tests for the /query API endpoint (api/main.py, api/routes/query.py)."""

from unittest.mock import AsyncMock

import httpx
import pytest

from agents.orchestrator import OrchestratorResult
from agents.planner import Plan
from agents.retriever import RetrievalResult
from api.main import app


@pytest.mark.asyncio
async def test_query_endpoint_returns_answer_with_citations() -> None:
    fake_orchestrator = AsyncMock()
    fake_orchestrator.ask.return_value = OrchestratorResult(
        question="what are the incoterms?",
        plan=Plan(needs_knowledge_search=True, search_query="incoterms", needs_tracking_lookup=False),
        retrieval=RetrievalResult(),
        answer="Incoterms define buyer/seller responsibilities.",
    )
    app.state.orchestrator = fake_orchestrator

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"question": "what are the incoterms?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Incoterms define buyer/seller responsibilities."
    assert body["needs_knowledge_search"] is True
    fake_orchestrator.ask.assert_awaited_once_with("what are the incoterms?")


@pytest.mark.asyncio
async def test_query_endpoint_requires_question_field() -> None:
    app.state.orchestrator = AsyncMock()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={})

    assert response.status_code == 422
