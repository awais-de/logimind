"""Unit tests for retrieval/hybrid.py."""

from unittest.mock import Mock

from retrieval.hybrid import HybridSearchClient
from retrieval.semantic import SearchResult

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": None,
    "page_number": 1,
}


def _result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=f"text {chunk_id}", score=score, **SOURCE_KWARGS)


def _make_client(semantic_results, keyword_results, reranked_results):
    fake_semantic = Mock()
    fake_semantic.search.return_value = semantic_results
    fake_keyword = Mock()
    fake_keyword.search.return_value = keyword_results
    fake_reranker = Mock()
    fake_reranker.rerank.return_value = reranked_results
    client = HybridSearchClient(
        semantic_client=fake_semantic, keyword_index=fake_keyword, reranker=fake_reranker
    )
    return client, fake_semantic, fake_keyword, fake_reranker


def test_search_merges_and_reranks_combined_candidates() -> None:
    semantic_results = [_result("c1", 0.9), _result("c2", 0.8)]
    keyword_results = [_result("c3", 5.0)]
    reranked = [_result("c3", 0.99), _result("c1", 0.5)]
    client, _, _, fake_reranker = _make_client(semantic_results, keyword_results, reranked)

    result = client.search("query", top_k=2)

    assert result == reranked
    rerank_call = fake_reranker.rerank.call_args
    assert rerank_call.args[0] == "query"
    assert {r.chunk_id for r in rerank_call.args[1]} == {"c1", "c2", "c3"}
    assert rerank_call.kwargs["top_k"] == 2


def test_search_deduplicates_chunks_present_in_both_result_sets() -> None:
    semantic_results = [_result("c1", 0.9)]
    keyword_results = [_result("c1", 5.0), _result("c2", 3.0)]
    client, _, _, fake_reranker = _make_client(semantic_results, keyword_results, [])

    client.search("query")

    candidates = fake_reranker.rerank.call_args.args[1]
    assert [r.chunk_id for r in candidates] == ["c1", "c2"]


def test_search_passes_candidate_pool_size_to_sub_searches() -> None:
    client, fake_semantic, fake_keyword, _ = _make_client([], [], [])

    client.search("query", candidate_pool_size=20)

    fake_semantic.search.assert_called_once_with("query", top_k=20)
    fake_keyword.search.assert_called_once_with("query", top_k=20)


def test_search_with_no_candidates_returns_empty_without_reranking() -> None:
    client, _, _, fake_reranker = _make_client([], [], [])

    result = client.search("query")

    assert result == []
    fake_reranker.rerank.assert_not_called()
