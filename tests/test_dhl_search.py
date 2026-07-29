"""Unit tests for agents/tools/dhl_search.py."""

from unittest.mock import Mock, patch

import agents.tools.dhl_search as dhl_search
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


def setup_function() -> None:
    dhl_search._client = None


def test_dhl_knowledge_search_returns_client_results() -> None:
    fake_client = Mock()
    fake_client.search.return_value = [_result("c1")]

    with patch.object(dhl_search, "HybridSearchClient", return_value=fake_client):
        results = dhl_search.dhl_knowledge_search("what are the customs rules", top_k=3)

    fake_client.search.assert_called_once_with("what are the customs rules", top_k=3)
    assert results == [_result("c1")]


def test_dhl_knowledge_search_uses_default_top_k() -> None:
    fake_client = Mock()
    fake_client.search.return_value = []

    with patch.object(dhl_search, "HybridSearchClient", return_value=fake_client):
        dhl_search.dhl_knowledge_search("query")

    fake_client.search.assert_called_once_with("query", top_k=dhl_search.DEFAULT_TOP_K)


def test_dhl_knowledge_search_reuses_cached_client() -> None:
    fake_client = Mock()
    fake_client.search.return_value = []

    with patch.object(dhl_search, "HybridSearchClient", return_value=fake_client) as mock_ctor:
        dhl_search.dhl_knowledge_search("query one")
        dhl_search.dhl_knowledge_search("query two")

    mock_ctor.assert_called_once()
