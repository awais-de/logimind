"""Unit tests for retrieval/cache.py."""

from unittest.mock import Mock

from retrieval.cache import SemanticCache


def _embedding_response(vector: list[float]) -> Mock:
    response = Mock()
    response.data = [Mock(embedding=vector)]
    return response


def test_get_returns_none_on_empty_cache_without_embedding_call() -> None:
    client = Mock()
    cache = SemanticCache(client=client)

    result = cache.get("what are the incoterms?")

    assert result is None
    client.embeddings.create.assert_not_called()


def test_set_then_get_identical_query_hits() -> None:
    client = Mock()
    client.embeddings.create.return_value = _embedding_response([1.0, 0.0])
    cache = SemanticCache(client=client, threshold=0.97)

    cache.set("what are the incoterms?", "cached answer")
    result = cache.get("what are the incoterms?")

    assert result == "cached answer"


def test_get_hits_for_similar_query_above_threshold() -> None:
    client = Mock()
    client.embeddings.create.side_effect = [
        _embedding_response([1.0, 0.0]),  # set()
        _embedding_response([0.99, 0.01]),  # get(), close but not identical
    ]
    cache = SemanticCache(client=client, threshold=0.9)

    cache.set("what are the incoterms?", "cached answer")
    result = cache.get("what are incoterms exactly?")

    assert result == "cached answer"


def test_get_misses_for_dissimilar_query_below_threshold() -> None:
    client = Mock()
    client.embeddings.create.side_effect = [
        _embedding_response([1.0, 0.0]),  # set()
        _embedding_response([0.0, 1.0]),  # get(), orthogonal
    ]
    cache = SemanticCache(client=client, threshold=0.9)

    cache.set("what are the incoterms?", "cached answer")
    result = cache.get("where is my package?")

    assert result is None


def test_get_matches_best_of_multiple_entries() -> None:
    client = Mock()
    client.embeddings.create.side_effect = [
        _embedding_response([1.0, 0.0]),  # set() entry A
        _embedding_response([0.0, 1.0]),  # set() entry B
        _embedding_response([0.0, 0.99]),  # get(), closest to entry B
    ]
    cache = SemanticCache(client=client, threshold=0.9)

    cache.set("query A", "answer A")
    cache.set("query B", "answer B")
    result = cache.get("query B, roughly")

    assert result == "answer B"
