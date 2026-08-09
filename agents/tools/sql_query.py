"""Read-only natural-language-to-SQL tool for RetrieverAgent.

PlannerAgent writes the SQL text itself (given the structured dataset's
schema in its system prompt) -- this module's job is purely to validate
that text is a safe, read-only SELECT and execute it, since the query
text is LLM-generated and shouldn't be trusted by default.
"""

import logging
import re
import sqlite3
from pathlib import Path

from data.structured.segment_revenue import STRUCTURED_DB_PATH

logger = logging.getLogger(__name__)

_DISALLOWED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH"
    r"|PRAGMA|VACUUM|REINDEX|GRANT|EXEC|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)
_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


class UnsafeQueryError(ValueError):
    """Raised when a generated SQL query isn't a safe, read-only SELECT."""


def _validate_read_only(query: str) -> None:
    """Reject anything that isn't a single, plain SELECT statement.

    Args:
        query: The SQL text to validate.

    Raises:
        UnsafeQueryError: The query is empty, isn't a SELECT, or contains
            a disallowed keyword (writes, schema changes, attaching
            another database file, pragmas, etc).
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        raise UnsafeQueryError("Query is empty")
    if not _SELECT_RE.match(stripped):
        raise UnsafeQueryError(f"Only SELECT queries are allowed, got: {query!r}")
    if _DISALLOWED_KEYWORDS.search(stripped):
        raise UnsafeQueryError(f"Query contains a disallowed keyword: {query!r}")


def run_sql_query(query: str, db_path: Path = STRUCTURED_DB_PATH) -> list[dict]:
    """Execute a read-only SQL query against the structured dataset.

    Validates the query is a single plain SELECT before opening the
    database in SQLite's read-only URI mode as a hard backstop --
    keyword filtering alone can be worked around, so the connection
    itself is also incapable of writing.

    Args:
        query: The SQL SELECT statement to execute.
        db_path: Path to the structured dataset's SQLite database.

    Returns:
        Result rows as a list of dicts, keyed by column name.

    Raises:
        UnsafeQueryError: The query isn't a safe, read-only SELECT.
        sqlite3.Error: The query is syntactically invalid, or references
            a table/column that doesn't exist.
    """
    _validate_read_only(query)

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(query).fetchall()]
    finally:
        connection.close()

    return rows
