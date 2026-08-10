"""Latency and token-cost tracking per agent step, stored in SQLite."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

METRICS_DB_PATH = Path(__file__).resolve().parents[1] / "monitoring" / "metrics.db"

# Per-million-token USD pricing. Sonnet 5 intro pricing ($2.00/$10.00) is
# active through 2026-08-31; reverts to the standard $3.00/$15.00 after.
MODEL_PRICING_PER_MILLION = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


class StepMetric(BaseModel):
    """Latency and cost for a single agent step within one query.

    Attributes:
        query_id: Identifier grouping steps from the same Orchestrator.ask()
            call.
        step: Which step this is ("planner", "retriever", "responder").
        latency_ms: Wall-clock time the step took.
        model: Model used for this step, if it made an LLM call.
        prompt_tokens: Input tokens billed, if applicable.
        completion_tokens: Output tokens billed, if applicable.
        cost_usd: Computed cost in USD, if applicable.
        timestamp: When the step completed, UTC.
    """

    query_id: str
    step: str
    latency_ms: float
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    timestamp: datetime


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Compute USD cost for a step's token usage.

    Args:
        model: Model ID, matched against MODEL_PRICING_PER_MILLION.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.

    Returns:
        Cost in USD, or None if the model isn't in the pricing table.
    """
    pricing = MODEL_PRICING_PER_MILLION.get(model)
    if pricing is None:
        return None
    return (
        prompt_tokens / 1_000_000 * pricing["input"]
        + completion_tokens / 1_000_000 * pricing["output"]
    )


def _init_db(db_path: Path) -> None:
    """Create the step_metrics table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS step_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                step TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                cost_usd REAL,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def record_step(metric: StepMetric, db_path: Path = METRICS_DB_PATH) -> None:
    """Persist one step's metrics to SQLite.

    Args:
        metric: The step metric to record.
        db_path: SQLite database file to write to.
    """
    _init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            INSERT INTO step_metrics
                (query_id, step, latency_ms, model, prompt_tokens,
                 completion_tokens, cost_usd, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric.query_id,
                metric.step,
                metric.latency_ms,
                metric.model,
                metric.prompt_tokens,
                metric.completion_tokens,
                metric.cost_usd,
                metric.timestamp.isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    logger.info(
        "Recorded metric: query=%s step=%s latency=%.0fms cost=%s",
        metric.query_id,
        metric.step,
        metric.latency_ms,
        f"${metric.cost_usd:.5f}" if metric.cost_usd is not None else "n/a",
    )
