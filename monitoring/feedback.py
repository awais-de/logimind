"""User feedback (thumbs up/down) logging, joined against query_log."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from monitoring.eval_loop import EVAL_DB_PATH

logger = logging.getLogger(__name__)

Vote = Literal["up", "down"]


class FeedbackVote(BaseModel):
    """A thumbs up/down vote on one query's answer.

    Attributes:
        query_id: The monitoring/eval_loop.py query_log row this vote is
            for.
        vote: "up" or "down".
        timestamp: When the vote was cast.
    """

    query_id: str
    vote: Vote
    timestamp: datetime


class FeedbackEntry(BaseModel):
    """A feedback vote joined with the query_log row it was cast on.

    Attributes:
        query_id: The voted-on query.
        vote: "up" or "down".
        timestamp: When the vote was cast.
        question: The original user question.
        answer: The answer that was voted on.
        context: Retrieved chunk text the answer was grounded in, if any.
    """

    query_id: str
    vote: Vote
    timestamp: datetime
    question: str
    answer: str
    context: list[str]


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                query_id TEXT PRIMARY KEY,
                vote TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def record_feedback(feedback: FeedbackVote, db_path: Path = EVAL_DB_PATH) -> None:
    """Persist a feedback vote, keyed to its query_log row.

    Args:
        feedback: The vote to record.
        db_path: SQLite database to write to. Shares
            monitoring/eval_loop.py's query_log table (same default file)
            so votes can be joined back to their original query.
    """
    _init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "INSERT OR REPLACE INTO feedback (query_id, vote, timestamp) VALUES (?, ?, ?)",
            (feedback.query_id, feedback.vote, feedback.timestamp.isoformat()),
        )
        connection.commit()
    finally:
        connection.close()
    logger.info("Recorded feedback: query=%s vote=%s", feedback.query_id, feedback.vote)


def get_disputed_queries(db_path: Path = EVAL_DB_PATH) -> list[FeedbackEntry]:
    """Fetch every downvoted query, joined with its question/answer/context.

    Meant to surface hard cases for review -- disputed answers can be
    folded into the curated ground-truth eval set instead of relying only
    on it.

    Args:
        db_path: SQLite database to read from. Assumes
            monitoring/eval_loop.py's query_log table already exists
            there (a vote can only be cast on an already-logged query).

    Returns:
        Downvoted queries, most recently voted first.
    """
    _init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT f.query_id, f.vote, f.timestamp, ql.question, ql.answer, ql.context
            FROM feedback f
            JOIN query_log ql ON ql.query_id = f.query_id
            WHERE f.vote = 'down'
            ORDER BY f.timestamp DESC
            """
        ).fetchall()
    finally:
        connection.close()

    return [
        FeedbackEntry(
            query_id=row[0],
            vote=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            question=row[3],
            answer=row[4],
            context=json.loads(row[5]),
        )
        for row in rows
    ]
