"""Ingestion lineage: run identity and content-hash tracking for stored chunks.

Used by data/ingestion/store.py to stamp every chunk written to SQLite/
Qdrant with which ingestion run produced it and when, and to detect
chunks whose content hasn't changed since their last write -- so
re-running ingestion against unchanged source PDFs is a no-op rather
than a rewrite.
"""

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class RunInfo(BaseModel):
    """Identity of one ingestion run, shared by every chunk it writes.

    Attributes:
        run_id: Unique identifier for this run.
        ingested_at: When this run happened.
    """

    run_id: str
    ingested_at: datetime

    @classmethod
    def new(cls) -> "RunInfo":
        """Start a fresh run: a new run_id, timestamped now (UTC)."""
        return cls(run_id=str(uuid.uuid4()), ingested_at=datetime.now(timezone.utc))


def content_hash(text: str) -> str:
    """Fingerprint a chunk's text, to detect unchanged content on a re-run.

    Args:
        text: The chunk's text.

    Returns:
        A hex-encoded SHA-256 digest of the text.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def get_chunk_lineage(chunk_id: str, db_path: Path) -> dict | None:
    """Look up which ingestion run last wrote a chunk, and when.

    Answers "when was this chunk last refreshed" against
    data/ingestion/store.py's chunks table.

    Args:
        chunk_id: The chunk to look up.
        db_path: Path to store.py's SQLite database file (its SQLITE_PATH).

    Returns:
        {"run_id": ..., "ingested_at": ...}, or None if the chunk isn't
        stored. Both values are None if the chunk predates lineage
        tracking (written before these columns existed).
    """
    connection = sqlite3.connect(db_path)
    try:
        table_exists = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if table_exists is None:
            return None
        try:
            row = connection.execute(
                "SELECT run_id, ingested_at FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            # A chunks table that predates lineage tracking has no
            # run_id/ingested_at columns at all.
            return None
    finally:
        connection.close()

    if row is None:
        return None
    return {"run_id": row[0], "ingested_at": row[1]}
