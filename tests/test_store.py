"""Unit tests for data/ingestion/store.py."""

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct

from data.ingestion.chunker import Chunk
from data.ingestion.embedder import EmbeddedChunk
from data.ingestion.store import (
    QdrantStore,
    _point_id,
    store_chunks,
    write_to_qdrant,
    write_to_sqlite,
)

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": "US",
    "publication_date": date(2024, 1, 1),
}


def _embedded_chunk(chunk_id: str, page_number: int = 1) -> EmbeddedChunk:
    chunk = Chunk(chunk_id=chunk_id, page_number=page_number, text=f"text for {chunk_id}", **SOURCE_KWARGS)
    return EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2, 0.3])


def test_point_id_is_stable_and_unique() -> None:
    assert _point_id("chunk-a") == _point_id("chunk-a")
    assert _point_id("chunk-a") != _point_id("chunk-b")


def _make_store() -> QdrantStore:
    return QdrantStore(url="http://localhost:6333", api_key="test-key")


def test_qdrant_store_ensure_collection_creates_when_missing() -> None:
    store = _make_store()
    store._client.get_collections = Mock(return_value=Mock(collections=[]))
    store._client.create_collection = Mock()

    store.ensure_collection(dim=1536)

    store._client.create_collection.assert_called_once()


def test_qdrant_store_ensure_collection_skips_when_present() -> None:
    store = _make_store()
    existing = Mock()
    existing.name = "logimind_chunks"
    store._client.get_collections = Mock(return_value=Mock(collections=[existing]))
    store._client.create_collection = Mock()

    store.ensure_collection(dim=1536)

    store._client.create_collection.assert_not_called()


_DUMMY_POINT = PointStruct(id=1, vector=[0.1], payload={})


def test_qdrant_store_upsert_retries_then_succeeds() -> None:
    store = _make_store()
    error = UnexpectedResponse(
        status_code=503, reason_phrase="unavailable", content=b"", headers={}
    )
    store._client.upsert = Mock(side_effect=[error, None])

    with patch("data.ingestion.store.time.sleep"):
        store.upsert([_DUMMY_POINT])

    assert store._client.upsert.call_count == 2


def test_qdrant_store_upsert_raises_after_max_retries() -> None:
    store = _make_store()
    error = UnexpectedResponse(
        status_code=503, reason_phrase="unavailable", content=b"", headers={}
    )
    store._client.upsert = Mock(side_effect=error)

    with patch("data.ingestion.store.time.sleep"), pytest.raises(UnexpectedResponse):
        store.upsert([_DUMMY_POINT])

    assert store._client.upsert.call_count == 3


def test_qdrant_store_upsert_batches_large_point_lists() -> None:
    store = _make_store()
    store._client.upsert = Mock()
    points = [PointStruct(id=i, vector=[0.1], payload={}) for i in range(450)]

    store.upsert(points)

    assert store._client.upsert.call_count == 3


def test_write_to_qdrant_ensures_collection_and_upserts() -> None:
    embedded = [_embedded_chunk("c1")]
    fake_store = Mock()

    write_to_qdrant(embedded, store=fake_store)

    fake_store.ensure_collection.assert_called_once_with(dim=3)
    fake_store.upsert.assert_called_once()
    points = fake_store.upsert.call_args[0][0]
    assert len(points) == 1
    assert points[0].payload["chunk_id"] == "c1"


def test_write_to_sqlite_creates_table_and_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    embedded = [_embedded_chunk("c1"), _embedded_chunk("c2", page_number=2)]

    write_to_sqlite(embedded, db_path=db_path)

    connection = sqlite3.connect(db_path)
    rows = connection.execute("SELECT chunk_id, page_number, text FROM chunks ORDER BY chunk_id").fetchall()
    connection.close()

    assert rows == [("c1", 1, "text for c1"), ("c2", 2, "text for c2")]


def test_write_to_sqlite_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    embedded = [_embedded_chunk("c1")]

    write_to_sqlite(embedded, db_path=db_path)
    write_to_sqlite(embedded, db_path=db_path)

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    connection.close()

    assert count == 1


def test_store_chunks_writes_to_both_backends() -> None:
    embedded = [_embedded_chunk("c1")]

    with patch("data.ingestion.store.write_to_qdrant") as mock_qdrant, patch(
        "data.ingestion.store.write_to_sqlite"
    ) as mock_sqlite:
        store_chunks(embedded)

    mock_qdrant.assert_called_once_with(embedded)
    mock_sqlite.assert_called_once_with(embedded)
