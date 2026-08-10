"""The /feedback endpoint: records a thumbs up/down vote on an answer."""

import logging
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models import FeedbackRequest
from monitoring.feedback import FeedbackVote, record_feedback

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/feedback", status_code=204)
async def feedback(payload: FeedbackRequest) -> None:
    """Record a feedback vote on a previously returned answer.

    Args:
        payload: The query_id being voted on and the vote itself.

    Raises:
        HTTPException: 500 if the vote couldn't be persisted.
    """
    vote = FeedbackVote(
        query_id=payload.query_id, vote=payload.vote, timestamp=datetime.now(timezone.utc)
    )
    try:
        record_feedback(vote)
    except sqlite3.Error as exc:
        logger.error("Failed to record feedback for query %s: %s", payload.query_id, exc)
        raise HTTPException(status_code=500, detail="Failed to record feedback") from exc
