"""Unit tests for retrieval/keyword.py."""

import sqlite3
from pathlib import Path

from retrieval.keyword import KeywordSearchIndex, _load_chunks_from_sqlite

CHUNKS = [
    {
        "chunk_id": "c1",
        "doc_id": "customs-doc",
        "doc_name": "Customs Guide",
        "category": "customs",
        "region": None,
        "page_number": 1,
        "text": "the customs guide covers duties and taxes on imports",
    },
    {
        "chunk_id": "c2",
        "doc_id": "rate-doc",
        "doc_name": "Rate Guide",
        "category": "rate_guide",
        "region": "US",
        "page_number": 3,
        "text": "rate guide for shipping packages internationally",
    },
    {
        "chunk_id": "c3",
        "doc_id": "restricted-doc",
        "doc_name": "Restricted Items Guide",
        "category": "restricted_items",
        "region": None,
        "page_number": 1,
        "text": "prohibited items include batteries and cannabis derivatives",
    },
]


def test_search_returns_relevant_chunk_first() -> None:
    index = KeywordSearchIndex(chunks=CHUNKS)

    results = index.search("what are the customs duties on imports", top_k=3)

    assert results[0].chunk_id == "c1"
    assert results[0].doc_name == "Customs Guide"
    assert results[0].score > 0


def test_search_drops_zero_score_results() -> None:
    index = KeywordSearchIndex(chunks=CHUNKS)

    results = index.search("customs duties", top_k=3)

    assert all(r.score > 0 for r in results)
    assert "c2" not in {r.chunk_id for r in results}


def test_search_respects_top_k() -> None:
    index = KeywordSearchIndex(chunks=CHUNKS)

    results = index.search("guide", top_k=1)

    assert len(results) <= 1


def test_search_on_empty_index_returns_empty_list() -> None:
    index = KeywordSearchIndex(chunks=[])

    assert index.search("anything") == []


def test_load_chunks_from_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE chunks (
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
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("c1", "doc-1", "Doc One", "test", None, None, 1, "some chunk text"),
    )
    connection.commit()
    connection.close()

    chunks = _load_chunks_from_sqlite(db_path)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "c1"
    assert chunks[0]["text"] == "some chunk text"
