"""Unit tests for agents/retriever.py."""

from unittest.mock import patch

from agents.planner import Plan
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
    plan = Plan(needs_knowledge_search=True, search_query="incoterms", needs_tracking_lookup=False)

    with patch("agents.retriever.dhl_knowledge_search", return_value=[_result("c1")]) as mock_search, \
         patch("agents.retriever.dhl_tracking_mock") as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_called_once_with("incoterms")
    mock_tracking.assert_not_called()
    assert result.search_results == [_result("c1")]
    assert result.tracking_info is None


def test_run_retriever_calls_tracking_when_needed() -> None:
    plan = Plan(needs_knowledge_search=False, needs_tracking_lookup=True, tracking_number="12345")

    with patch("agents.retriever.dhl_knowledge_search") as mock_search, \
         patch("agents.retriever.dhl_tracking_mock", return_value={"status": "In Transit"}) as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_not_called()
    mock_tracking.assert_called_once_with("12345")
    assert result.search_results == []
    assert result.tracking_info == {"status": "In Transit"}


def test_run_retriever_calls_both_when_both_needed() -> None:
    plan = Plan(
        needs_knowledge_search=True,
        search_query="customs",
        needs_tracking_lookup=True,
        tracking_number="999",
    )

    with patch("agents.retriever.dhl_knowledge_search", return_value=[_result("c1")]) as mock_search, \
         patch("agents.retriever.dhl_tracking_mock", return_value={"status": "Delivered"}) as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_called_once_with("customs")
    mock_tracking.assert_called_once_with("999")
    assert result.search_results == [_result("c1")]
    assert result.tracking_info == {"status": "Delivered"}


def test_run_retriever_calls_neither_when_plan_says_no() -> None:
    plan = Plan(needs_knowledge_search=False, needs_tracking_lookup=False)

    with patch("agents.retriever.dhl_knowledge_search") as mock_search, \
         patch("agents.retriever.dhl_tracking_mock") as mock_tracking:
        result = run_retriever(plan)

    mock_search.assert_not_called()
    mock_tracking.assert_not_called()
    assert result.search_results == []
    assert result.tracking_info is None


def test_run_retriever_skips_search_if_flag_true_but_query_missing() -> None:
    plan = Plan(needs_knowledge_search=True, search_query=None, needs_tracking_lookup=False)

    with patch("agents.retriever.dhl_knowledge_search") as mock_search:
        result = run_retriever(plan)

    mock_search.assert_not_called()
    assert result.search_results == []
