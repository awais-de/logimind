"""Cross-encoder re-ranking of candidate search results."""

import logging

from sentence_transformers import CrossEncoder

from retrieval.semantic import SearchResult

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Re-ranks candidate chunks against a query using a cross-encoder."""

    def __init__(self, model_name: str = MODEL_NAME, model: CrossEncoder | None = None) -> None:
        """Load the cross-encoder model.

        Args:
            model_name: Sentence-transformers cross-encoder model to load.
            model: A pre-built cross-encoder to use instead of loading
                model_name. Mainly for tests.
        """
        self._model = model if model is not None else CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Re-score and re-sort candidate results against the query.

        Args:
            query: The original natural-language query.
            results: Candidate results to re-rank, e.g. the union of
                semantic and keyword search hits.
            top_k: If given, only the top_k results are kept after
                re-ranking. Defaults to returning all results, re-sorted.

        Returns:
            Results re-sorted by descending cross-encoder score, with each
            result's score field replaced by the cross-encoder score.
        """
        if not results:
            return []

        pairs = [(query, result.text) for result in results]
        cross_scores = self._model.predict(pairs)

        reranked = [
            result.model_copy(update={"score": float(score)})
            for result, score in zip(results, cross_scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked
