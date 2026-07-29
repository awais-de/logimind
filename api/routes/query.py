"""The /query endpoint: routes a question through the agent pipeline."""

import logging

from fastapi import APIRouter, Request

from agents.orchestrator import Orchestrator
from api.models import QueryRequest, QueryResponse
from api.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter()

QUERY_RATE_LIMIT = "10/minute"


@router.post("/query", response_model=QueryResponse)
@limiter.limit(QUERY_RATE_LIMIT)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Answer a user question via the full agent pipeline.

    Rate limited (10/minute per client IP) since each call spends real
    OpenAI/Anthropic API budget.

    Args:
        payload: The user's question.
        request: The incoming request, used to access the shared
            Orchestrator stored on app.state. Must be named exactly
            "request" -- slowapi's rate-limit decorator looks it up by
            that literal name, not by type.

    Returns:
        The synthesized answer with citations.
    """
    orchestrator: Orchestrator = request.app.state.orchestrator
    result = await orchestrator.ask(payload.question)
    return QueryResponse.from_orchestrator_result(result)
