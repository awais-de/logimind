"""Unit tests for agents/retriever.py."""

from unittest.mock import patch

import pytest

from agents.planner import Plan, Step
from agents.retriever import run_retriever
from retrieval.semantic import SearchResult

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": None,
    "page_number": 1,
}


def _result(chunk_id: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=f"text {chunk_id}", score=0.9, **SOURCE_KWARGS)


def test_run_retriever_calls_knowledge_search_when_needed() -> None:
    plan = Plan(steps=[Step(tool="knowledge_search", search_query="incoterms")])

    with patch("agents.retriever.dhl_knowledge_search", return_value=[_result("c1")]) as mock_search, \
         patch("agents.retriever.dhl_tracking_mock") as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_called_once_with("incoterms")
    mock_tracking.assert_not_called()
    assert result.search_results == [_result("c1")]
    assert result.tracking_info is None


def test_run_retriever_calls_tracking_when_needed() -> None:
    plan = Plan(steps=[Step(tool="tracking_lookup", tracking_number="12345")])

    with patch("agents.retriever.dhl_knowledge_search") as mock_search, \
         patch("agents.retriever.dhl_tracking_mock", return_value={"status": "In Transit"}) as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_not_called()
    mock_tracking.assert_called_once_with("12345")
    assert result.search_results == []
    assert result.tracking_info == {"status": "In Transit"}


def test_run_retriever_calls_both_when_both_needed() -> None:
    plan = Plan(
        steps=[
            Step(tool="knowledge_search", search_query="customs"),
            Step(tool="tracking_lookup", tracking_number="999"),
        ]
    )

    with patch("agents.retriever.dhl_knowledge_search", return_value=[_result("c1")]) as mock_search, \
         patch("agents.retriever.dhl_tracking_mock", return_value={"status": "Delivered"}) as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_called_once_with("customs")
    mock_tracking.assert_called_once_with("999")
    assert result.search_results == [_result("c1")]
    assert result.tracking_info == {"status": "Delivered"}


def test_run_retriever_calls_neither_when_plan_has_no_steps() -> None:
    plan = Plan()

    with patch("agents.retriever.dhl_knowledge_search") as mock_search, \
         patch("agents.retriever.dhl_tracking_mock") as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_not_called()
    mock_tracking.assert_not_called()
    assert result.search_results == []
    assert result.tracking_info is None


def test_run_retriever_skips_search_step_with_no_query() -> None:
    plan = Plan(steps=[Step(tool="knowledge_search", search_query=None)])

    with patch("agents.retriever.dhl_knowledge_search") as mock_search:
        result = run_retriever(plan)

    mock_search.assert_not_called()
    assert result.search_results == []


def test_run_retriever_resolves_placeholder_from_earlier_tracking_step() -> None:
    plan = Plan(
        steps=[
            Step(tool="tracking_lookup", tracking_number="1234567890"),
            Step(tool="knowledge_search", search_query="customs rules for {{step_1.destination}}"),
        ]
    )

    with patch(
        "agents.retriever.dhl_tracking_mock", return_value={"destination": "Bonn, DE"}
    ) as mock_tracking, patch(
        "agents.retriever.dhl_knowledge_search", return_value=[_result("c1")]
    ) as mock_search:
        result = run_retriever(plan)

    mock_tracking.assert_called_once_with("1234567890")
    mock_search.assert_called_once_with("customs rules for Bonn, DE")
    assert result.tracking_info == {"destination": "Bonn, DE"}
    assert result.search_results == [_result("c1")]


def test_run_retriever_raises_when_placeholder_references_future_step() -> None:
    plan = Plan(steps=[Step(tool="knowledge_search", search_query="rules for {{step_2.destination}}")])

    with patch("agents.retriever.dhl_knowledge_search") as mock_search:
        with pytest.raises(ValueError, match="step_2"):
            run_retriever(plan)

    mock_search.assert_not_called()


def test_run_retriever_raises_when_placeholder_field_missing() -> None:
    plan = Plan(
        steps=[
            Step(tool="tracking_lookup", tracking_number="123"),
            Step(tool="knowledge_search", search_query="rules for {{step_1.nonexistent_field}}"),
        ]
    )

    with patch("agents.retriever.dhl_tracking_mock", return_value={"destination": "Bonn, DE"}):
        with pytest.raises(ValueError, match="nonexistent_field"):
            run_retriever(plan)
