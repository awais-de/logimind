"""Write embedded chunks to Qdrant (vectors) and SQLite (metadata index)."""

import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import Distance, PointStruct, VectorParams

from data.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "logimind_chunks"
EMBEDDING_DIM = 1536
POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "logimind.chunks")
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0

SQLITE_PATH = Path(__file__).resolve().parents[2] / "data" / "metadata.db"


def _point_id(chunk_id: str) -> str:
    """Derive a stable Qdrant point UUID from a chunk_id."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


class QdrantStore:
    """Wraps the Qdrant client: collection setup and retried writes."""

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        """Initialize the store.

        Args:
            url: Qdrant cluster URL. Falls back to the QDRANT_URL env var.
            api_key: Qdrant API key. Falls back to the QDRANT_API_KEY env var.
        """
        self._client = QdrantClient(
            url=url or os.environ["QDRANT_URL"],
            api_key=api_key or os.environ.get("QDRANT_API_KEY"),
        )

    def ensure_collection(self, dim: int = EMBEDDING_DIM) -> None:
        """Create the chunks collection if it doesn't already exist.

        Args:
            dim: Embedding vector dimensionality.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection %s", COLLECTION_NAME)

    def upsert(self, points: list[PointStruct]) -> None:
        """Upsert points into the collection, retrying transient failures.

        Args:
            points: Points to upsert.

        Raises:
            qdrant_client.http.exceptions.ApiException: If the upsert fails
                after all retries.
        """
        last_error: ApiException | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self._client.upsert(collection_name=COLLECTION_NAME, points=points)
                return
            except ApiException as exc:
                last_error = exc
                wait = BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "Qdrant upsert failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise last_error


def _to_point(embedded: EmbeddedChunk) -> PointStruct:
    """Convert an EmbeddedChunk into a Qdrant point."""
    chunk = embedded.chunk
    return PointStruct(
        id=_point_id(chunk.chunk_id),
        vector=embedded.embedding,
        payload={
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_name": chunk.doc_name,
            "category": chunk.category,
            "region": chunk.region,
            "publication_date": chunk.publication_date.isoformat()
            if chunk.publication_date
            else None,
            "page_number": chunk.page_number,
            "text": chunk.text,
        },
    )


def write_to_qdrant(
    embedded_chunks: list[EmbeddedChunk], store: QdrantStore | None = None
) -> None:
    """Write embedded chunks into Qdrant.

    Args:
        embedded_chunks: Chunks with their embedding vectors.
        store: QdrantStore to use. Defaults to a new QdrantStore.
    """
    if not embedded_chunks:
        return
    if store is None:
        store = QdrantStore()

    store.ensure_collection(dim=len(embedded_chunks[0].embedding))
    points = [_to_point(embedded) for embedded in embedded_chunks]
    store.upsert(points)
    logger.info("Upserted %d points into Qdrant collection %s", len(points), COLLECTION_NAME)


def write_to_sqlite(embedded_chunks: list[EmbeddedChunk], db_path: Path = SQLITE_PATH) -> None:
    """Write chunk metadata into a SQLite index.

    Args:
        embedded_chunks: Chunks with their embedding vectors. Only metadata
            is written; vectors live in Qdrant.
        db_path: Path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                category TEXT NOT NULL,
                region TEXT,
                publication_date TEXT,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, doc_name, category, region,
                 publication_date, page_number, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    embedded.chunk.chunk_id,
                    embedded.chunk.doc_id,
                    embedded.chunk.doc_name,
                    embedded.chunk.category,
                    embedded.chunk.region,
                    embedded.chunk.publication_date.isoformat()
                    if embedded.chunk.publication_date
                    else None,
                    embedded.chunk.page_number,
                    embedded.chunk.text,
                )
                for embedded in embedded_chunks
            ],
        )
        connection.commit()
    finally:
        connection.close()
    logger.info("Wrote %d rows into SQLite metadata index at %s", len(embedded_chunks), db_path)


def store_chunks(embedded_chunks: list[EmbeddedChunk]) -> None:
    """Write embedded chunks to both Qdrant and the SQLite metadata index.

    Args:
        embedded_chunks: Chunks with their embedding vectors.
    """
    write_to_qdrant(embedded_chunks)
    write_to_sqlite(embedded_chunks)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from data.ingestion.chunker import chunk_all
    from data.ingestion.embedder import embed_chunks

    loaded_chunks = chunk_all()
    embedded_chunks = embed_chunks(loaded_chunks)
    store_chunks(embedded_chunks)
