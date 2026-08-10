"""Unit tests for the /feedback API endpoint."""

import sqlite3
from unittest.mock import patch

import httpx
import pytest

from api.main import app


@pytest.mark.asyncio
async def test_feedback_endpoint_records_vote() -> None:
    with patch("api.routes.feedback.record_feedback") as mock_record:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/feedback", json={"query_id": "q1", "vote": "up"})

    assert response.status_code == 204
    mock_record.assert_called_once()
    recorded_vote = mock_record.call_args.args[0]
    assert recorded_vote.query_id == "q1"
    assert recorded_vote.vote == "up"


@pytest.mark.asyncio
async def test_feedback_endpoint_rejects_invalid_vote() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/feedback", json={"query_id": "q1", "vote": "sideways"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_feedback_endpoint_returns_500_on_db_error() -> None:
    with patch("api.routes.feedback.record_feedback", side_effect=sqlite3.OperationalError("locked")):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/feedback", json={"query_id": "q1", "vote": "down"})

    assert response.status_code == 500
