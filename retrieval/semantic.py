"""Semantic vector search over the Qdrant chunk collection."""

import logging
import os
import time

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import ScoredPoint

from data.ingestion.embedder import EmbeddingClient
from data.ingestion.store import COLLECTION_NAME

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


class SearchResult(BaseModel):
    """A single semantic search hit.

    Attributes:
        chunk_id: Identifier of the matched chunk.
        doc_id: Source document identifier.
        doc_name: Source document title.
        category: Source document category.
        region: Source document region, if any.
        page_number: Page the chunk was taken from.
        text: The chunk's text.
        score: Similarity score from Qdrant (higher is more similar).
    """

    chunk_id: str
    doc_id: str
    doc_name: str
    category: str
    region: str | None
    page_number: int
    text: str
    score: float


def _to_result(hit: ScoredPoint) -> SearchResult:
    """Convert a Qdrant scored point into a SearchResult."""
    payload = hit.payload or {}
    return SearchResult(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        doc_name=payload["doc_name"],
        category=payload["category"],
        region=payload.get("region"),
        page_number=payload["page_number"],
        text=payload["text"],
        score=hit.score,
    )


class SemanticSearchClient:
    """Embeds queries and searches the Qdrant chunk collection, with retry."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            url: Qdrant cluster URL. Falls back to the QDRANT_URL env var.
            api_key: Qdrant API key. Falls back to the QDRANT_API_KEY env var.
            embedding_client: Client used to embed queries. Defaults to a
                new EmbeddingClient.
        """
        self._client = QdrantClient(
            url=url or os.environ["QDRANT_URL"],
            api_key=api_key or os.environ.get("QDRANT_API_KEY"),
        )
        self._embedding_client = embedding_client or EmbeddingClient()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Search for chunks semantically similar to a query.

        Args:
            query: Natural-language query text.
            top_k: Number of results to return.

        Returns:
            Search results ordered by descending similarity score.

        Raises:
            qdrant_client.http.exceptions.ApiException: If the query fails
                after all retries.
        """
        vector = self._embedding_client.embed_batch([query])[0]

        last_error: ApiException | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.query_points(
                    collection_name=COLLECTION_NAME, query=vector, limit=top_k
                )
                return [_to_result(hit) for hit in response.points]
            except ApiException as exc:
                last_error = exc
                wait = BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "Qdrant search failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise last_error
