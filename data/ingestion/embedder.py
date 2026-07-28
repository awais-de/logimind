"""Generate embeddings for chunks via the OpenAI embeddings API."""

import logging
import time

from openai import APIError, OpenAI
from pydantic import BaseModel

from data.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


class EmbeddedChunk(BaseModel):
    """A chunk paired with its embedding vector.

    Attributes:
        chunk: The source chunk.
        embedding: The chunk text's embedding vector.
    """

    chunk: Chunk
    embedding: list[float]


class EmbeddingClient:
    """Thin wrapper around the OpenAI embeddings API with retry."""

    def __init__(self, api_key: str | None = None, model: str = EMBEDDING_MODEL) -> None:
        """Initialize the client.

        Args:
            api_key: OpenAI API key. Falls back to the OPENAI_API_KEY
                environment variable if not given.
            model: Embedding model to use.
        """
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, retrying transient failures.

        Args:
            texts: Texts to embed, in order.

        Returns:
            Embedding vectors, in the same order as texts.

        Raises:
            openai.APIError: If the request fails after all retries.
        """
        last_error: APIError | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in response.data]
            except APIError as exc:
                last_error = exc
                wait = BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise last_error


def embed_chunks(
    chunks: list[Chunk],
    client: EmbeddingClient | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """Embed a list of chunks in batches.

    Args:
        chunks: Chunks to embed.
        client: Embedding client to use. Defaults to a new EmbeddingClient.
        batch_size: Number of chunks to embed per API call.

    Returns:
        One EmbeddedChunk per input chunk, in the same order.
    """
    if client is None:
        client = EmbeddingClient()

    embedded: list[EmbeddedChunk] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = client.embed_batch([chunk.text for chunk in batch])
        embedded.extend(
            EmbeddedChunk(chunk=chunk, embedding=vector)
            for chunk, vector in zip(batch, vectors)
        )
    return embedded
