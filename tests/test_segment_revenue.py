"""Unit tests for data/structured/segment_revenue.py."""

import sqlite3
from pathlib import Path

from data.structured.segment_revenue import load_segment_revenue


def test_load_creates_tables_and_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "structured.db"

    load_segment_revenue(db_path=db_path)

    connection = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    connection.close()

    assert {"segments", "segment_revenue", "business_units", "business_unit_revenue"} <= tables


def test_load_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "structured.db"

    load_segment_revenue(db_path=db_path)
    load_segment_revenue(db_path=db_path)

    connection = sqlite3.connect(db_path)
    segment_count = connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    revenue_count = connection.execute("SELECT COUNT(*) FROM segment_revenue").fetchone()[0]
    connection.close()

    assert segment_count == 5
    assert revenue_count == 10


def test_segment_revenue_matches_known_totals(tmp_path: Path) -> None:
    db_path = tmp_path / "structured.db"
    load_segment_revenue(db_path=db_path)

    connection = sqlite3.connect(db_path)
    express_2024 = connection.execute(
        "SELECT revenue_eur_millions FROM segment_revenue "
        "JOIN segments ON segments.id = segment_revenue.segment_id "
        "WHERE segments.name = 'Express' AND fiscal_year = 2024"
    ).fetchone()[0]
    connection.close()

    assert express_2024 == 24511


def test_business_units_sum_to_their_segment_total(tmp_path: Path) -> None:
    db_path = tmp_path / "structured.db"
    load_segment_revenue(db_path=db_path)

    connection = sqlite3.connect(db_path)
    segment_total = connection.execute(
        "SELECT revenue_eur_millions FROM segment_revenue "
        "JOIN segments ON segments.id = segment_revenue.segment_id "
        "WHERE segments.name = 'Post & Parcel Germany' AND fiscal_year = 2024"
    ).fetchone()[0]
    business_unit_sum = connection.execute(
        "SELECT SUM(revenue_eur_millions) FROM business_unit_revenue "
        "JOIN business_units ON business_units.id = business_unit_revenue.business_unit_id "
        "JOIN segments ON segments.id = business_units.segment_id "
        "WHERE segments.name = 'Post & Parcel Germany' AND fiscal_year = 2024"
    ).fetchone()[0]
    connection.close()

    assert business_unit_sum == segment_total


def test_business_unit_foreign_keys_reference_existing_segments(tmp_path: Path) -> None:
    db_path = tmp_path / "structured.db"
    load_segment_revenue(db_path=db_path)

    connection = sqlite3.connect(db_path)
    orphans = connection.execute(
        "SELECT COUNT(*) FROM business_units "
        "LEFT JOIN segments ON segments.id = business_units.segment_id "
        "WHERE segments.id IS NULL"
    ).fetchone()[0]
    connection.close()

    assert orphans == 0
