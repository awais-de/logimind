"""Unit tests for retrieval/reranker.py."""

from unittest.mock import Mock

from retrieval.reranker import Reranker
from retrieval.semantic import SearchResult

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": None,
    "page_number": 1,
}


def _result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, score=score, **SOURCE_KWARGS)


def test_rerank_resorts_by_cross_encoder_score() -> None:
    results = [_result("c1", "low relevance", 0.9), _result("c2", "high relevance", 0.1)]
    fake_model = Mock()
    fake_model.predict.return_value = [0.2, 0.8]

    reranker = Reranker(model=fake_model)
    reranked = reranker.rerank("query", results)

    assert [r.chunk_id for r in reranked] == ["c2", "c1"]
    assert reranked[0].score == 0.8


def test_rerank_respects_top_k() -> None:
    results = [_result("c1", "a", 0.5), _result("c2", "b", 0.5), _result("c3", "c", 0.5)]
    fake_model = Mock()
    fake_model.predict.return_value = [0.1, 0.9, 0.5]

    reranker = Reranker(model=fake_model)
    reranked = reranker.rerank("query", results, top_k=2)

    assert [r.chunk_id for r in reranked] == ["c2", "c3"]


def test_rerank_preserves_metadata_and_only_replaces_score() -> None:
    results = [_result("c1", "some text", 0.9)]
    fake_model = Mock()
    fake_model.predict.return_value = [3.5]

    reranker = Reranker(model=fake_model)
    reranked = reranker.rerank("query", results)

    assert reranked[0].chunk_id == "c1"
    assert reranked[0].doc_name == "Test Document"
    assert reranked[0].text == "some text"
    assert reranked[0].score == 3.5


def test_rerank_on_empty_results_returns_empty_without_calling_model() -> None:
    fake_model = Mock()

    reranker = Reranker(model=fake_model)
    reranked = reranker.rerank("query", [])

    assert reranked == []
    fake_model.predict.assert_not_called()
