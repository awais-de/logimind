"""Unit tests for agents/responder.py."""

from unittest.mock import AsyncMock, Mock

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import RequestUsage

from agents.model_client import build_claude_client
from agents.responder import _format_context, build_responder_agent, run_responder
from agents.retriever import RetrievalResult
from retrieval.semantic import SearchResult

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": None,
    "page_number": 3,
}


def _result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, score=0.9, **SOURCE_KWARGS)


def test_build_responder_agent_returns_named_assistant_agent() -> None:
    client = build_claude_client(api_key="dummy-test-key")

    agent = build_responder_agent(client)

    assert isinstance(agent, AssistantAgent)
    assert agent.name == "ResponseAgent"


def test_format_context_includes_question_and_excerpts() -> None:
    retrieval = RetrievalResult(search_results=[_result("c1", "some relevant text")])

    context = _format_context("what are the incoterms?", retrieval)

    assert "what are the incoterms?" in context
    assert "Test Document, p.3" in context
    assert "some relevant text" in context


def test_format_context_includes_tracking_info() -> None:
    retrieval = RetrievalResult(tracking_info={"status": "In Transit", "origin": "Bonn, DE"})

    context = _format_context("where is my package?", retrieval)

    assert "status: In Transit" in context
    assert "origin: Bonn, DE" in context


def test_format_context_handles_empty_retrieval() -> None:
    retrieval = RetrievalResult()

    context = _format_context("what's the weather?", retrieval)

    assert "No relevant document excerpts were found." in context
    assert "No tracking information was requested or found." in context


@pytest.mark.asyncio
async def test_run_responder_returns_agent_text_and_usage() -> None:
    fake_message = Mock()
    fake_message.content = "Prohibited items include batteries (Prohibited Items Guide, p.1)."
    fake_message.models_usage = RequestUsage(prompt_tokens=300, completion_tokens=80)
    fake_result = Mock()
    fake_result.messages = [fake_message]

    fake_agent = Mock()
    fake_agent.run = AsyncMock(return_value=fake_result)

    retrieval = RetrievalResult(search_results=[_result("c1", "batteries are prohibited")])
    answer, usage = await run_responder(fake_agent, "what's prohibited?", retrieval)

    fake_agent.run.assert_awaited_once()
    called_context = fake_agent.run.call_args.kwargs["task"]
    assert "what's prohibited?" in called_context
    assert answer == "Prohibited items include batteries (Prohibited Items Guide, p.1)."
    assert usage.prompt_tokens == 300
    assert usage.completion_tokens == 80
