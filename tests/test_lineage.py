"""Unit tests for data/ingestion/lineage.py."""

import sqlite3
from pathlib import Path

from data.ingestion.lineage import RunInfo, content_hash, get_chunk_lineage


def test_run_info_new_generates_unique_ids() -> None:
    first = RunInfo.new()
    second = RunInfo.new()

    assert first.run_id != second.run_id


def test_content_hash_is_deterministic() -> None:
    assert content_hash("some chunk text") == content_hash("some chunk text")


def test_content_hash_differs_for_different_text() -> None:
    assert content_hash("chunk a") != content_hash("chunk b")


def test_get_chunk_lineage_returns_none_when_db_missing(tmp_path: Path) -> None:
    assert get_chunk_lineage("c1", tmp_path / "does_not_exist.db") is None


def test_get_chunk_lineage_returns_none_when_table_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE unrelated (id TEXT)")
    connection.commit()
    connection.close()

    assert get_chunk_lineage("c1", db_path) is None


def test_get_chunk_lineage_returns_none_for_pre_lineage_table(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT, doc_name TEXT, category TEXT,
            region TEXT, publication_date TEXT, page_number INTEGER, text TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO chunks VALUES ('c1', 'd1', 'Doc', 'cat', NULL, NULL, 1, 'text')"
    )
    connection.commit()
    connection.close()

    assert get_chunk_lineage("c1", db_path) is None


def test_get_chunk_lineage_returns_run_info_when_present(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT, doc_name TEXT, category TEXT,
            region TEXT, publication_date TEXT, page_number INTEGER, text TEXT,
            run_id TEXT, ingested_at TEXT, content_hash TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO chunks VALUES ('c1', 'd1', 'Doc', 'cat', NULL, NULL, 1, 'text', "
        "'run-abc', '2026-08-02T00:00:00+00:00', 'hash')"
    )
    connection.commit()
    connection.close()

    lineage = get_chunk_lineage("c1", db_path)

    assert lineage == {"run_id": "run-abc", "ingested_at": "2026-08-02T00:00:00+00:00"}


def test_get_chunk_lineage_returns_none_for_unknown_chunk(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, run_id TEXT, ingested_at TEXT
        )
        """
    )
    connection.commit()
    connection.close()

    assert get_chunk_lineage("nonexistent", db_path) is None
