"""BM25 keyword search over the chunk metadata index."""

import logging
import sqlite3
from pathlib import Path

import bm25s

from data.ingestion.store import SQLITE_PATH
from retrieval.semantic import DEFAULT_TOP_K, SearchResult

logger = logging.getLogger(__name__)


def _load_chunks_from_sqlite(db_path: Path = SQLITE_PATH) -> list[dict]:
    """Load chunk metadata rows from the SQLite index."""
    connection = sqlite3.connect(db_path)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT chunk_id, doc_id, doc_name, category, region, page_number, text FROM chunks"
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


class KeywordSearchIndex:
    """BM25 keyword search index built from chunk metadata."""

    def __init__(self, chunks: list[dict] | None = None, db_path: Path = SQLITE_PATH) -> None:
        """Build the BM25 index.

        Args:
            chunks: Chunk metadata dicts (chunk_id, doc_id, doc_name,
                category, region, page_number, text) to index. Defaults to
                loading every chunk from the SQLite metadata index.
            db_path: SQLite database to load chunks from, if chunks is None.
        """
        if chunks is None:
            chunks = _load_chunks_from_sqlite(db_path)

        self._chunks = chunks
        self._retriever = bm25s.BM25(corpus=chunks)
        if chunks:
            corpus_tokens = bm25s.tokenize(
                [chunk["text"] for chunk in chunks], stopwords="en", show_progress=False
            )
            self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Search for chunks matching a query by keyword overlap.

        Args:
            query: Natural-language query text.
            top_k: Number of results to return.

        Returns:
            Search results with a positive BM25 score, ordered by
            descending score. Chunks with no keyword overlap are dropped
            rather than padding out to top_k.
        """
        if not self._chunks:
            return []

        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results, scores = self._retriever.retrieve(
            query_tokens, k=min(top_k, len(self._chunks)), show_progress=False
        )
        return [
            SearchResult(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                doc_name=chunk["doc_name"],
                category=chunk["category"],
                region=chunk["region"],
                page_number=chunk["page_number"],
                text=chunk["text"],
                score=float(score),
            )
            for chunk, score in zip(results[0], scores[0])
            if score > 0
        ]
