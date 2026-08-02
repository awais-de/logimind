"""Unit tests for agents/orchestrator.py."""

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from autogen_core.models import RequestUsage

from agents.model_client import build_claude_client
from agents.orchestrator import Orchestrator
from agents.planner import Plan
from agents.retriever import RetrievalResult
from retrieval.semantic import SearchResult


def _make_orchestrator(tmp_path: Path, model_client=None) -> Orchestrator:
    client = model_client or build_claude_client(api_key="dummy-test-key")
    return Orchestrator(
        model_client=client,
        metrics_db_path=tmp_path / "metrics.db",
        eval_db_path=tmp_path / "metrics.db",
    )


def test_orchestrator_builds_named_agents(tmp_path: Path) -> None:
    orchestrator = _make_orchestrator(tmp_path)

    assert orchestrator._planner.name == "PlannerAgent"
    assert orchestrator._responder.name == "ResponseAgent"


@pytest.mark.asyncio
async def test_ask_runs_planner_then_retriever_then_responder(tmp_path: Path) -> None:
    orchestrator = _make_orchestrator(tmp_path)

    fake_plan = Plan(needs_knowledge_search=True, search_query="incoterms", needs_tracking_lookup=False)
    fake_retrieval = RetrievalResult()

    with patch(
        "agents.orchestrator.run_planner", new=AsyncMock(return_value=(fake_plan, None))
    ) as mock_planner, patch(
        "agents.orchestrator.run_retriever", return_value=fake_retrieval
    ) as mock_retriever, patch(
        "agents.orchestrator.run_responder", new=AsyncMock(return_value=("final answer", None))
    ) as mock_responder:
        result = await orchestrator.ask("what are the incoterms?")

    mock_planner.assert_awaited_once_with(orchestrator._planner, "what are the incoterms?")
    mock_retriever.assert_called_once_with(fake_plan)
    mock_responder.assert_awaited_once_with(orchestrator._responder, "what are the incoterms?", fake_retrieval)

    assert result.question == "what are the incoterms?"
    assert result.plan == fake_plan
    assert result.retrieval == fake_retrieval
    assert result.answer == "final answer"


@pytest.mark.asyncio
async def test_ask_records_metrics_for_all_three_steps(tmp_path: Path) -> None:
    orchestrator = _make_orchestrator(tmp_path)
    fake_plan = Plan(needs_knowledge_search=False, needs_tracking_lookup=False)
    planner_usage = RequestUsage(prompt_tokens=100, completion_tokens=20)
    responder_usage = RequestUsage(prompt_tokens=200, completion_tokens=50)

    with patch(
        "agents.orchestrator.run_planner", new=AsyncMock(return_value=(fake_plan, planner_usage))
    ), patch(
        "agents.orchestrator.run_retriever", return_value=RetrievalResult()
    ), patch(
        "agents.orchestrator.run_responder",
        new=AsyncMock(return_value=("answer", responder_usage)),
    ):
        await orchestrator.ask("hello")

    connection = sqlite3.connect(tmp_path / "metrics.db")
    rows = connection.execute(
        "SELECT step, model, prompt_tokens, completion_tokens, cost_usd FROM step_metrics ORDER BY id"
    ).fetchall()
    connection.close()

    assert [r[0] for r in rows] == ["planner", "retriever", "responder"]
    assert rows[0][1] == orchestrator._model
    assert rows[0][2:4] == (100, 20)
    assert rows[0][4] is not None
    assert rows[1][1] is None  # retriever isn't LLM-backed, no model/cost
    assert rows[1][4] is None
    assert rows[2][2:4] == (200, 50)


@pytest.mark.asyncio
async def test_ask_logs_query_sample_for_eval(tmp_path: Path) -> None:
    orchestrator = _make_orchestrator(tmp_path)
    fake_plan = Plan(needs_knowledge_search=True, search_query="incoterms", needs_tracking_lookup=False)
    fake_result = SearchResult(
        chunk_id="c1", text="Incoterms define shipping responsibilities.", score=0.9,
        doc_id="doc1", doc_name="Incoterms Guide", category="incoterms", region=None, page_number=1,
    )
    fake_retrieval = RetrievalResult(search_results=[fake_result])

    with patch(
        "agents.orchestrator.run_planner", new=AsyncMock(return_value=(fake_plan, None))
    ), patch(
        "agents.orchestrator.run_retriever", return_value=fake_retrieval
    ), patch(
        "agents.orchestrator.run_responder", new=AsyncMock(return_value=("final answer", None))
    ):
        await orchestrator.ask("what are the incoterms?")

    connection = sqlite3.connect(tmp_path / "metrics.db")
    row = connection.execute("SELECT question, answer, context FROM query_log").fetchone()
    connection.close()

    assert row[0] == "what are the incoterms?"
    assert row[1] == "final answer"
    assert "Incoterms define shipping responsibilities." in row[2]


@pytest.mark.asyncio
async def test_close_closes_model_client(tmp_path: Path) -> None:
    fake_client = Mock()
    fake_client.close = AsyncMock()
    fake_client.model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    }

    orchestrator = _make_orchestrator(tmp_path, model_client=fake_client)
    await orchestrator.close()

    fake_client.close.assert_awaited_once()
