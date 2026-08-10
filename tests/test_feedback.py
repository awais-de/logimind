"""Unit tests for monitoring/feedback.py."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from monitoring.eval_loop import QuerySample, log_query_sample
from monitoring.feedback import FeedbackVote, get_disputed_queries, record_feedback


def test_record_feedback_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    vote = FeedbackVote(query_id="q1", vote="up", timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc))

    record_feedback(vote, db_path=db_path)

    connection = sqlite3.connect(db_path)
    row = connection.execute("SELECT query_id, vote FROM feedback").fetchone()
    connection.close()

    assert row == ("q1", "up")


def test_record_feedback_is_idempotent_per_query(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"

    record_feedback(
        FeedbackVote(query_id="q1", vote="up", timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc)),
        db_path=db_path,
    )
    record_feedback(
        FeedbackVote(query_id="q1", vote="down", timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc)),
        db_path=db_path,
    )

    connection = sqlite3.connect(db_path)
    rows = connection.execute("SELECT query_id, vote FROM feedback").fetchall()
    connection.close()

    assert rows == [("q1", "down")]


def test_get_disputed_queries_joins_query_log(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    log_query_sample(
        QuerySample(
            query_id="q1", question="what are the incoterms?", answer="wrong answer",
            context=["some context"], timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        db_path=db_path,
    )
    log_query_sample(
        QuerySample(
            query_id="q2", question="where is my package?", answer="correct answer",
            timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        db_path=db_path,
    )
    record_feedback(
        FeedbackVote(query_id="q1", vote="down", timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc)),
        db_path=db_path,
    )
    record_feedback(
        FeedbackVote(query_id="q2", vote="up", timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc)),
        db_path=db_path,
    )

    disputed = get_disputed_queries(db_path=db_path)

    assert len(disputed) == 1
    assert disputed[0].query_id == "q1"
    assert disputed[0].question == "what are the incoterms?"
    assert disputed[0].answer == "wrong answer"
    assert disputed[0].context == ["some context"]


def test_get_disputed_queries_empty_when_no_downvotes(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    log_query_sample(
        QuerySample(
            query_id="q1", question="q", answer="a", timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc)
        ),
        db_path=db_path,
    )
    record_feedback(
        FeedbackVote(query_id="q1", vote="up", timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc)),
        db_path=db_path,
    )

    assert get_disputed_queries(db_path=db_path) == []
