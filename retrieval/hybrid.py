"""Combine semantic and keyword search, then re-rank the merged results."""

import logging

from retrieval.keyword import KeywordSearchIndex
from retrieval.reranker import Reranker
from retrieval.semantic import DEFAULT_TOP_K, SearchResult, SemanticSearchClient

logger = logging.getLogger(__name__)

CANDIDATE_POOL_SIZE = 15


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Drop duplicate chunks, keeping the first occurrence of each."""
    seen: set[str] = set()
    deduplicated = []
    for result in results:
        if result.chunk_id not in seen:
            seen.add(result.chunk_id)
            deduplicated.append(result)
    return deduplicated


class HybridSearchClient:
    """Combines semantic + keyword search and re-ranks the merged results.

    This is the main retrieval entry point Layer 3 agents call.
    """

    def __init__(
        self,
        semantic_client: SemanticSearchClient | None = None,
        keyword_index: KeywordSearchIndex | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Wire up the three retrieval components.

        Args:
            semantic_client: Client for Qdrant vector search. Defaults to a
                new SemanticSearchClient.
            keyword_index: BM25 keyword index. Defaults to a new
                KeywordSearchIndex built from the SQLite metadata index.
            reranker: Cross-encoder reranker. Defaults to a new Reranker.
        """
        self._semantic_client = semantic_client or SemanticSearchClient()
        self._keyword_index = keyword_index or KeywordSearchIndex()
        self._reranker = reranker or Reranker()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    ) -> list[SearchResult]:
        """Search using both semantic and keyword retrieval, then re-rank.

        Args:
            query: Natural-language query text.
            top_k: Number of final results to return after re-ranking.
            candidate_pool_size: Number of results to pull from each of
                semantic and keyword search before deduplication and
                re-ranking.

        Returns:
            The top_k results, re-ranked by the cross-encoder.
        """
        semantic_results = self._semantic_client.search(query, top_k=candidate_pool_size)
        keyword_results = self._keyword_index.search(query, top_k=candidate_pool_size)

        candidates = _deduplicate(semantic_results + keyword_results)
        if not candidates:
            return []

        return self._reranker.rerank(query, candidates, top_k=top_k)
