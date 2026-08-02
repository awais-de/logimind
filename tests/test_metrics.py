"""Unit tests for monitoring/metrics.py."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from monitoring.metrics import StepMetric, compute_cost, record_step


def test_compute_cost_known_model() -> None:
    cost = compute_cost("claude-sonnet-5", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert cost == 2.00 + 10.00


def test_compute_cost_unknown_model_returns_none() -> None:
    assert compute_cost("some-other-model", prompt_tokens=1000, completion_tokens=1000) is None


def test_record_step_writes_row(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    metric = StepMetric(
        query_id="q1",
        step="planner",
        latency_ms=123.4,
        model="claude-sonnet-5",
        prompt_tokens=100,
        completion_tokens=20,
        cost_usd=0.0004,
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    record_step(metric, db_path=db_path)

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT query_id, step, latency_ms, model, prompt_tokens, completion_tokens, cost_usd FROM step_metrics"
    ).fetchone()
    connection.close()

    assert row == ("q1", "planner", 123.4, "claude-sonnet-5", 100, 20, 0.0004)


def test_record_step_handles_missing_usage(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    metric = StepMetric(
        query_id="q2",
        step="retriever",
        latency_ms=50.0,
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    record_step(metric, db_path=db_path)

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT model, prompt_tokens, completion_tokens, cost_usd FROM step_metrics"
    ).fetchone()
    connection.close()

    assert row == (None, None, None, None)


def test_record_step_accumulates_multiple_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    for step in ["planner", "retriever", "responder"]:
        record_step(
            StepMetric(
                query_id="q3",
                step=step,
                latency_ms=10.0,
                timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
            ),
            db_path=db_path,
        )

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM step_metrics").fetchone()[0]
    connection.close()

    assert count == 3
