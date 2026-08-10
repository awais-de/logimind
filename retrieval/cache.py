"""In-memory semantic cache: short-circuits repeated/near-duplicate queries."""

import logging
import math

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SIMILARITY_THRESHOLD = 0.97


class _CacheEntry(BaseModel):
    query: str
    embedding: list[float]
    value: str


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """An in-process, embedding-similarity cache for arbitrary string values.

    Not tied to any particular caller's data shape -- a caller stores
    whatever string payload it wants (e.g. a serialized orchestrator
    result) keyed by a natural-language query, and gets it back for a
    later query that's semantically close enough, not just
    character-identical.

    Process-lifetime only: entries are held in memory and lost on
    restart. A persistent backend (e.g. SQLite, matching the rest of this
    project's storage) is a natural extension if surviving restarts ever
    matters more than implementation simplicity.
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        """Set up the cache.

        Args:
            client: OpenAI client used for embeddings. Defaults to a new
                OpenAI().
            threshold: Minimum cosine similarity for an incoming query to
                count as a hit against a previously cached query.
            embedding_model: Embedding model used to compare queries.
        """
        self._client = client or OpenAI()
        self._threshold = threshold
        self._embedding_model = embedding_model
        self._entries: list[_CacheEntry] = []

    def _embed(self, query: str) -> list[float]:
        response = self._client.embeddings.create(model=self._embedding_model, input=[query])
        return response.data[0].embedding

    def get(self, query: str) -> str | None:
        """Look up the cached value for a semantically similar query.

        Args:
            query: The incoming query to check against cached entries.

        Returns:
            The cached value if a prior query is similar enough (cosine
            similarity >= threshold), else None. Returns None without an
            embedding call if the cache is empty.
        """
        if not self._entries:
            return None

        embedding = self._embed(query)
        best_entry, best_similarity = None, -1.0
        for entry in self._entries:
            similarity = _cosine_similarity(embedding, entry.embedding)
            if similarity > best_similarity:
                best_entry, best_similarity = entry, similarity

        if best_entry is not None and best_similarity >= self._threshold:
            logger.info(
                "Cache hit: %r matched %r (similarity=%.4f)", query, best_entry.query, best_similarity
            )
            return best_entry.value

        return None

    def set(self, query: str, value: str) -> None:
        """Cache a value for a query.

        Args:
            query: The query to cache under.
            value: The value to return for future semantically similar
                queries.
        """
        embedding = self._embed(query)
        self._entries.append(_CacheEntry(query=query, embedding=embedding, value=value))
