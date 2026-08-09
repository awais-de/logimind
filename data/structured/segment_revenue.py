"""Curated DHL Group segment revenue dataset for natural-language-to-SQL queries.

Source: DHL Group 2024 Annual Report, note "11 Revenue by business unit",
p.182 (data/raw/dhl-annual-report-2024.pdf). Figures are in EUR millions,
for fiscal years 2023 and 2024. Two divisions break down further into
business units in the source table; the rest report a single segment
total, which the relational shape below preserves.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

STRUCTURED_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "structured.db"

SEGMENTS: list[tuple[int, str]] = [
    (1, "Express"),
    (2, "Global Forwarding, Freight"),
    (3, "Supply Chain"),
    (4, "eCommerce"),
    (5, "Post & Parcel Germany"),
]

SEGMENT_REVENUE: list[tuple[int, int, float]] = [
    (1, 2023, 24322), (1, 2024, 24511),
    (2, 2023, 18031), (2, 2024, 18403),
    (3, 2023, 16814), (3, 2024, 17584),
    (4, 2023, 6174), (4, 2024, 6787),
    (5, 2023, 16402), (5, 2024, 16893),
]

BUSINESS_UNITS: list[tuple[int, int, str]] = [
    (1, 2, "Global Forwarding"),
    (2, 2, "Freight"),
    (3, 5, "Post Germany"),
    (4, 5, "Parcel Germany"),
    (5, 5, "International"),
    (6, 5, "Other"),
]

BUSINESS_UNIT_REVENUE: list[tuple[int, int, float]] = [
    (1, 2023, 13981), (1, 2024, 14352),
    (2, 2023, 4050), (2, 2024, 4051),
    (3, 2023, 7505), (3, 2024, 7319),
    (4, 2023, 6747), (4, 2024, 7316),
    (5, 2023, 1999), (5, 2024, 2076),
    (6, 2023, 151), (6, 2024, 182),
]


def load_segment_revenue(db_path: Path = STRUCTURED_DB_PATH) -> None:
    """Load the curated segment revenue dataset into a SQLite database.

    Creates four tables: segments, segment_revenue, business_units, and
    business_unit_revenue -- business_units and business_unit_revenue
    hold the finer breakdown for the two segments the source table splits
    further, joined back to segments via segment_id. Idempotent: tables
    are created if missing, and rows are inserted with INSERT OR REPLACE
    so re-running this doesn't duplicate data.

    Args:
        db_path: Path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_revenue (
                segment_id INTEGER NOT NULL REFERENCES segments(id),
                fiscal_year INTEGER NOT NULL,
                revenue_eur_millions REAL NOT NULL,
                PRIMARY KEY (segment_id, fiscal_year)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS business_units (
                id INTEGER PRIMARY KEY,
                segment_id INTEGER NOT NULL REFERENCES segments(id),
                name TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS business_unit_revenue (
                business_unit_id INTEGER NOT NULL REFERENCES business_units(id),
                fiscal_year INTEGER NOT NULL,
                revenue_eur_millions REAL NOT NULL,
                PRIMARY KEY (business_unit_id, fiscal_year)
            )
            """
        )

        connection.executemany("INSERT OR REPLACE INTO segments (id, name) VALUES (?, ?)", SEGMENTS)
        connection.executemany(
            "INSERT OR REPLACE INTO segment_revenue (segment_id, fiscal_year, revenue_eur_millions) VALUES (?, ?, ?)",
            SEGMENT_REVENUE,
        )
        connection.executemany(
            "INSERT OR REPLACE INTO business_units (id, segment_id, name) VALUES (?, ?, ?)", BUSINESS_UNITS
        )
        connection.executemany(
            "INSERT OR REPLACE INTO business_unit_revenue "
            "(business_unit_id, fiscal_year, revenue_eur_millions) VALUES (?, ?, ?)",
            BUSINESS_UNIT_REVENUE,
        )
        connection.commit()
    finally:
        connection.close()
    logger.info("Loaded segment revenue dataset into %s", db_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_segment_revenue()
