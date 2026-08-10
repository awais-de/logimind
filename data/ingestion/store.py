"""Write embedded chunks to Qdrant (vectors) and SQLite (metadata index).

Every write carries ingestion lineage (data/ingestion/lineage.py) -- a
run_id shared by every chunk written in the same store_chunks() call, an
ingested_at timestamp, and a content_hash used to make re-running
ingestion against unchanged source PDFs a no-op: a chunk whose text
hasn't changed since its last write is skipped entirely (row untouched,
no bumped run_id/timestamp) rather than rewritten with new lineage on
every run.
"""

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
from data.ingestion.lineage import RunInfo, content_hash

logger = logging.getLogger(__name__)

COLLECTION_NAME = "logimind_chunks"
EMBEDDING_DIM = 1536
POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "logimind.chunks")
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0
UPSERT_BATCH_SIZE = 200

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
        """Upsert points into the collection in batches, retrying transient
        failures within each batch.

        Points are sent in batches of UPSERT_BATCH_SIZE rather than all at
        once, since Qdrant's REST API rejects request payloads over 32MB.

        Args:
            points: Points to upsert.

        Raises:
            qdrant_client.http.exceptions.ApiException: If a batch fails
                after all retries.
        """
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            self._upsert_batch(batch)

    def _upsert_batch(self, points: list[PointStruct]) -> None:
        """Upsert a single batch of points, retrying transient failures."""
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


def _to_point(embedded: EmbeddedChunk, run: RunInfo) -> PointStruct:
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
            "run_id": run.run_id,
            "ingested_at": run.ingested_at.isoformat(),
        },
    )


def write_to_qdrant(
    embedded_chunks: list[EmbeddedChunk],
    store: QdrantStore | None = None,
    run: RunInfo | None = None,
) -> None:
    """Write embedded chunks into Qdrant.

    Args:
        embedded_chunks: Chunks with their embedding vectors.
        store: QdrantStore to use. Defaults to a new QdrantStore.
        run: Identity of this ingestion run, stamped onto every point's
            payload. Defaults to a fresh RunInfo.new().
    """
    if not embedded_chunks:
        return
    if store is None:
        store = QdrantStore()
    run = run or RunInfo.new()

    store.ensure_collection(dim=len(embedded_chunks[0].embedding))
    points = [_to_point(embedded, run) for embedded in embedded_chunks]
    store.upsert(points)
    logger.info(
        "Upserted %d points into Qdrant collection %s (run_id=%s)",
        len(points), COLLECTION_NAME, run.run_id,
    )


def _init_sqlite_schema(connection: sqlite3.Connection) -> None:
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
            text TEXT NOT NULL,
            run_id TEXT,
            ingested_at TEXT,
            content_hash TEXT
        )
        """
    )
    # A chunks table created before lineage tracking existed won't have
    # these columns yet; add them without losing existing rows. Harmless
    # no-op (duplicate column) on a fresh table.
    for column in ("run_id", "ingested_at", "content_hash"):
        try:
            connection.execute(f"ALTER TABLE chunks ADD COLUMN {column} TEXT")
        except sqlite3.OperationalError:
            pass


def write_to_sqlite(
    embedded_chunks: list[EmbeddedChunk],
    db_path: Path = SQLITE_PATH,
    run: RunInfo | None = None,
) -> int:
    """Write chunk metadata into a SQLite index.

    A chunk whose text is byte-identical to what's already stored for its
    chunk_id is skipped entirely -- re-running ingestion against
    unchanged source PDFs touches zero rows, rather than rewriting every
    row with a new run_id/timestamp on every run.

    Args:
        embedded_chunks: Chunks with their embedding vectors. Only metadata
            is written; vectors live in Qdrant.
        db_path: Path to the SQLite database file.
        run: Identity of this ingestion run, stamped onto every row
            actually written. Defaults to a fresh RunInfo.new().

    Returns:
        Number of rows actually written (changed or new) -- 0 if every
        chunk's content already matched what was stored.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    run = run or RunInfo.new()
    ingested_at_str = run.ingested_at.isoformat()

    connection = sqlite3.connect(db_path)
    try:
        _init_sqlite_schema(connection)

        written = 0
        for embedded in embedded_chunks:
            chunk = embedded.chunk
            chunk_hash = content_hash(chunk.text)
            existing = connection.execute(
                "SELECT content_hash FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
            ).fetchone()
            if existing is not None and existing[0] == chunk_hash:
                continue

            connection.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, doc_name, category, region, publication_date,
                     page_number, text, run_id, ingested_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.doc_name,
                    chunk.category,
                    chunk.region,
                    chunk.publication_date.isoformat() if chunk.publication_date else None,
                    chunk.page_number,
                    chunk.text,
                    run.run_id,
                    ingested_at_str,
                    chunk_hash,
                ),
            )
            written += 1

        connection.commit()
    finally:
        connection.close()
    logger.info(
        "Wrote %d/%d rows into SQLite metadata index at %s (run_id=%s)",
        written, len(embedded_chunks), db_path, run.run_id,
    )
    return written


def store_chunks(embedded_chunks: list[EmbeddedChunk]) -> None:
    """Write embedded chunks to both Qdrant and the SQLite metadata index.

    Both stores are stamped with the same run identity, so one ingestion
    run has one consistent run_id/ingested_at across both.

    Args:
        embedded_chunks: Chunks with their embedding vectors.
    """
    run = RunInfo.new()
    write_to_qdrant(embedded_chunks, run=run)
    write_to_sqlite(embedded_chunks, run=run)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    from data.ingestion.chunker import chunk_all
    from data.ingestion.embedder import embed_chunks

    loaded_chunks = chunk_all()
    embedded_chunks = embed_chunks(loaded_chunks)
    store_chunks(embedded_chunks)
