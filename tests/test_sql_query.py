"""Unit tests for agents/tools/sql_query.py."""

import sqlite3
from pathlib import Path

import pytest

from agents.tools.sql_query import UnsafeQueryError, run_sql_query
from data.structured.segment_revenue import load_segment_revenue


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "structured.db"
    load_segment_revenue(db_path=path)
    return path


def test_executes_simple_select(db_path: Path) -> None:
    rows = run_sql_query("SELECT name FROM segments ORDER BY name", db_path=db_path)

    assert {row["name"] for row in rows} == {
        "Express",
        "Global Forwarding, Freight",
        "Supply Chain",
        "eCommerce",
        "Post & Parcel Germany",
    }


def test_executes_aggregate_query(db_path: Path) -> None:
    rows = run_sql_query(
        "SELECT SUM(revenue_eur_millions) AS total FROM segment_revenue WHERE fiscal_year = 2024",
        db_path=db_path,
    )

    assert rows == [{"total": 24511 + 18403 + 17584 + 6787 + 16893}]


def test_executes_join_query(db_path: Path) -> None:
    rows = run_sql_query(
        "SELECT segments.name, segment_revenue.revenue_eur_millions FROM segment_revenue "
        "JOIN segments ON segments.id = segment_revenue.segment_id "
        "WHERE segments.name = 'Express' AND fiscal_year = 2024",
        db_path=db_path,
    )

    assert rows == [{"name": "Express", "revenue_eur_millions": 24511}]


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE segments",
        "DELETE FROM segments",
        "UPDATE segments SET name = 'x'",
        "INSERT INTO segments (id, name) VALUES (99, 'x')",
        "ALTER TABLE segments ADD COLUMN x TEXT",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA table_info(segments)",
        "SELECT 1; DROP TABLE segments;",
        "",
        "   ",
    ],
)
def test_rejects_unsafe_queries_before_execution(db_path: Path, query: str) -> None:
    with pytest.raises(UnsafeQueryError):
        run_sql_query(query, db_path=db_path)

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    connection.close()
    assert count == 5


def test_invalid_sql_syntax_raises_sqlite_error(db_path: Path) -> None:
    with pytest.raises(sqlite3.Error):
        run_sql_query("SELECT this is not valid sql", db_path=db_path)


def test_unknown_table_raises_sqlite_error(db_path: Path) -> None:
    with pytest.raises(sqlite3.Error):
        run_sql_query("SELECT * FROM nonexistent_table", db_path=db_path)
