"""RAG search tool for RetrieverAgent: wraps hybrid retrieval."""

import logging

from retrieval.hybrid import HybridSearchClient
from retrieval.semantic import DEFAULT_TOP_K, SearchResult

logger = logging.getLogger(__name__)

_client: HybridSearchClient | None = None


def _get_client() -> HybridSearchClient:
    """Lazily create and cache the HybridSearchClient.

    Building a HybridSearchClient loads the cross-encoder model and the
    BM25 index, so it's created once and reused across tool calls rather
    than per call.
    """
    global _client
    if _client is None:
        _client = HybridSearchClient()
    return _client


def dhl_knowledge_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
    """Search DHL's ingested knowledge base for chunks relevant to a query.

    This is the tool RetrieverAgent calls to answer questions using DHL's
    public operational documents (rate guides, customs guidelines, annual
    reports, etc).

    Args:
        query: Natural-language question or search query.
        top_k: Number of relevant chunks to return.

    Returns:
        Relevant chunks with source citations (doc_name, page_number),
        ordered by relevance.
    """
    return _get_client().search(query, top_k=top_k)
