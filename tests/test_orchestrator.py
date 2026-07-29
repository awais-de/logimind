"""Unit tests for agents/orchestrator.py."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.model_client import build_claude_client
from agents.orchestrator import Orchestrator
from agents.planner import Plan
from agents.retriever import RetrievalResult


def test_orchestrator_builds_named_agents() -> None:
    client = build_claude_client(api_key="dummy-test-key")

    orchestrator = Orchestrator(model_client=client)

    assert orchestrator._planner.name == "PlannerAgent"
    assert orchestrator._responder.name == "ResponseAgent"


@pytest.mark.asyncio
async def test_ask_runs_planner_then_retriever_then_responder() -> None:
    client = build_claude_client(api_key="dummy-test-key")
    orchestrator = Orchestrator(model_client=client)

    fake_plan = Plan(needs_knowledge_search=True, search_query="incoterms", needs_tracking_lookup=False)
    fake_retrieval = RetrievalResult()

    with patch("agents.orchestrator.run_planner", new=AsyncMock(return_value=fake_plan)) as mock_planner, \
         patch("agents.orchestrator.run_retriever", return_value=fake_retrieval) as mock_retriever, \
         patch("agents.orchestrator.run_responder", new=AsyncMock(return_value="final answer")) as mock_responder:
        result = await orchestrator.ask("what are the incoterms?")

    mock_planner.assert_awaited_once_with(orchestrator._planner, "what are the incoterms?")
    mock_retriever.assert_called_once_with(fake_plan)
    mock_responder.assert_awaited_once_with(orchestrator._responder, "what are the incoterms?", fake_retrieval)

    assert result.question == "what are the incoterms?"
    assert result.plan == fake_plan
    assert result.retrieval == fake_retrieval
    assert result.answer == "final answer"


@pytest.mark.asyncio
async def test_close_closes_model_client() -> None:
    fake_client = Mock()
    fake_client.close = AsyncMock()
    fake_client.model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    }

    orchestrator = Orchestrator(model_client=fake_client)
    await orchestrator.close()

    fake_client.close.assert_awaited_once()
