"""Unit tests for retrieval/semantic.py."""

from unittest.mock import Mock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models.models import QueryResponse
from qdrant_client.models import ScoredPoint

from retrieval.semantic import SemanticSearchClient


def _scored_point(chunk_id: str, score: float) -> ScoredPoint:
    return ScoredPoint(
        id=chunk_id,
        version=0,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "doc_id": "test-doc",
            "doc_name": "Test Document",
            "category": "test",
            "region": "US",
            "page_number": 1,
            "text": f"text for {chunk_id}",
        },
    )


def _make_client() -> tuple[SemanticSearchClient, Mock]:
    fake_embedder = Mock()
    fake_embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]
    client = SemanticSearchClient(
        url="http://localhost:6333", api_key="test-key", embedding_client=fake_embedder
    )
    return client, fake_embedder


def test_search_embeds_query_and_returns_results() -> None:
    client, fake_embedder = _make_client()
    response = QueryResponse(points=[_scored_point("c1", 0.9), _scored_point("c2", 0.8)])
    client._client.query_points = Mock(return_value=response)

    results = client.search("what are the customs rules", top_k=2)

    fake_embedder.embed_batch.assert_called_once_with(["what are the customs rules"])
    assert [r.chunk_id for r in results] == ["c1", "c2"]
    assert [r.score for r in results] == [0.9, 0.8]
    assert results[0].doc_name == "Test Document"


def test_search_retries_then_succeeds() -> None:
    client, _ = _make_client()
    error = UnexpectedResponse(status_code=503, reason_phrase="unavailable", content=b"", headers={})
    response = QueryResponse(points=[_scored_point("c1", 0.9)])
    client._client.query_points = Mock(side_effect=[error, response])

    with patch("retrieval.semantic.time.sleep"):
        results = client.search("query")

    assert len(results) == 1
    assert client._client.query_points.call_count == 2


def test_search_raises_after_max_retries() -> None:
    client, _ = _make_client()
    error = UnexpectedResponse(status_code=503, reason_phrase="unavailable", content=b"", headers={})
    client._client.query_points = Mock(side_effect=error)

    with patch("retrieval.semantic.time.sleep"), pytest.raises(UnexpectedResponse):
        client.search("query")

    assert client._client.query_points.call_count == 3
