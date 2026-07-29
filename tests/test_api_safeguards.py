"""Unit tests for the /query endpoint's rate limit and question-length guard."""

from unittest.mock import AsyncMock

import httpx
import pytest

from agents.orchestrator import OrchestratorResult
from agents.planner import Plan
from agents.retriever import RetrievalResult
from api.main import app
from api.models import MAX_QUESTION_LENGTH


def _fake_result(question: str) -> OrchestratorResult:
    return OrchestratorResult(
        question=question,
        plan=Plan(needs_knowledge_search=False, needs_tracking_lookup=False),
        retrieval=RetrievalResult(),
        answer="answer",
    )


@pytest.mark.asyncio
async def test_query_rejects_question_over_max_length() -> None:
    app.state.orchestrator = AsyncMock()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query", json={"question": "x" * (MAX_QUESTION_LENGTH + 1)}
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_empty_question() -> None:
    app.state.orchestrator = AsyncMock()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"question": ""})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_accepts_question_at_max_length() -> None:
    fake_orchestrator = AsyncMock()
    fake_orchestrator.ask.return_value = _fake_result("x" * MAX_QUESTION_LENGTH)
    app.state.orchestrator = fake_orchestrator
    app.state.limiter.reset()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"question": "x" * MAX_QUESTION_LENGTH},
            headers={"X-Forwarded-For": "10.0.0.1"},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_rate_limits_after_threshold() -> None:
    fake_orchestrator = AsyncMock()
    fake_orchestrator.ask.return_value = _fake_result("question")
    app.state.orchestrator = fake_orchestrator
    app.state.limiter.reset()

    transport = httpx.ASGITransport(app=app)
    headers = {"X-Forwarded-For": "10.0.0.2"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        statuses = []
        for _ in range(11):
            response = await client.post("/query", json={"question": "question"}, headers=headers)
            statuses.append(response.status_code)

    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429


@pytest.mark.asyncio
async def test_query_rate_limit_is_per_client_ip() -> None:
    fake_orchestrator = AsyncMock()
    fake_orchestrator.ask.return_value = _fake_result("question")
    app.state.orchestrator = fake_orchestrator
    app.state.limiter.reset()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(10):
            await client.post(
                "/query", json={"question": "question"}, headers={"X-Forwarded-For": "10.0.0.3"}
            )
        exhausted = await client.post(
            "/query", json={"question": "question"}, headers={"X-Forwarded-For": "10.0.0.3"}
        )
        other_client = await client.post(
            "/query", json={"question": "question"}, headers={"X-Forwarded-For": "10.0.0.4"}
        )

    assert exhausted.status_code == 429
    assert other_client.status_code == 200
